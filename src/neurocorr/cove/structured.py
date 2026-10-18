"""
Structured covariance estimators: factor analysis, sparse, and sparse plus latent.

Python equivalents of ``+cove/factor.m`` and ``+cove/lvglasso.m`` in
https://github.com/atlab/cov-est (MIT License, (c) 2014 Tolias Lab):

- ``factor_analysis`` maximizes the same Gaussian likelihood as the EM in ``factor.m``, by
  L-BFGS-B over the log uniquenesses with the loadings in closed form (profile likelihood).
  It converges where the EM's 200-cycle cap stops short.
- ``sparse_latent`` solves the problem of ``lvglasso.m`` with gglasso's ADMM, to convergence:
  minimize tr(Sigma (S - L)) - log det(S - L) + alpha sum|S_ij| + beta tr(L), L psd. As in the original,
  the L1 penalty includes the diagonal of S. With ``latent=False`` it is the graphical lasso.
- ``sparse_latent_capped`` is the rank-capped variant (cov-est method 91), which no library
  solves: a numpy port of the original's proximal ADMM with the rank of L capped.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from scipy import linalg, optimize

# gglasso's ADMM: penalty on the diagonal as cov-est, converged as checked against regain and cvxpy
ADMM_TOLERANCE = {"tol": 1e-7, "rtol": 1e-4, "max_iter": 5000}


def factor_analysis(C: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximum-likelihood factor analysis of a covariance matrix: C ≈ L L' + diag(psi).

    Returns the loadings L (p x k) and the uniquenesses psi (p,).
    """
    p = C.shape[0]
    d = np.diag(C)
    k = int(k)
    if k == 0:
        return np.zeros((p, 0)), d.copy()

    def unpack(z):
        psi = np.exp(z)
        s = np.sqrt(psi)
        lam, U = linalg.eigh(C / np.outer(s, s))
        lam, U = lam[::-1], U[:, ::-1]
        top = np.maximum(lam[:k], 1.0)
        return psi, (s[:, None] * U[:, :k]) * np.sqrt(top - 1.0), lam, top

    def negative_log_likelihood(z):
        psi, L, lam, top = unpack(z)
        f = (
            np.sum(np.log(psi))
            + np.sum(np.log(top))
            + np.sum(lam[:k] / top)
            + np.sum(lam[k:])
        )
        model = L @ L.T + np.diag(psi)
        inverse = np.linalg.inv(model)
        return f, np.diag(inverse @ (model - C) @ inverse) * psi

    result = optimize.minimize(
        negative_log_likelihood,
        np.log(d / 2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(np.log(1e-8 * di), np.log(di)) for di in d],
        options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-15, "gtol": 1e-10},
    )
    psi, L, _, _ = unpack(result.x)
    return L, psi


def sparse_latent(
    Sigma: np.ndarray, alpha: float, beta: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse (and, with beta, low-rank) decomposition of the precision matrix of Sigma.

    Returns S, the sparse component, and L, the low-rank component (zero without beta), so that
    inv(S - L) is the estimate of Sigma.
    """
    from gglasso.solver.single_admm_solver import ADMM_SGL

    p = Sigma.shape[0]
    latent = beta is not None
    with contextlib.redirect_stdout(io.StringIO()):
        solution, _ = ADMM_SGL(
            Sigma,
            float(alpha),
            np.eye(p),
            latent=latent,
            mu1=float(beta) if latent else None,
            off_diagonal_l1=False,
            **ADMM_TOLERANCE,
        )
    S = (solution["Theta"] + solution["Theta"].T) / 2
    L = (solution["L"] + solution["L"].T) / 2 if latent else np.zeros((p, p))
    return S, L


def sparse_latent_capped(
    Sigma: np.ndarray,
    alpha: float,
    max_rank: int,
    beta: float = 0.001,
    tolerance: float = 1e-5,
    max_iter: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse plus low-rank precision with the rank of L capped (cov-est method 91).

    The proximal ADMM of ``lvglasso.m`` (Ma, Xue & Zou, 2013): mu starts at p and shrinks by a
    quarter every 6 iterations down to 0.01, tau = 0.6, and the run stops when the relative
    primal residual falls below ``tolerance``. Each update keeps the top ``max_rank``
    eigenvalues of L.
    """
    p = Sigma.shape[0]
    rank = min(int(max_rank), p - 1)
    mu, tau = float(p), 0.6
    S, L, Lam = np.eye(p), np.zeros((p, p)), np.zeros((p, p))
    for iteration in range(1, max_iter + 1):
        d, U = np.linalg.eigh(mu * Sigma - mu * Lam - S + L)
        R = (U * ((-d + np.sqrt(d * d + 4 * mu)) / 2)) @ U.T
        step = S - L - R + mu * Lam
        S = S - tau * step
        S = np.sign(S) * np.maximum(np.abs(S) - tau * mu * alpha, 0)
        S = (S + S.T) / 2
        d, U = np.linalg.eigh(L + tau * step)
        d[: p - rank] = 0
        L = (U * np.maximum(0, d - tau * mu * beta)) @ U.T
        L = (L + L.T) / 2
        Lam = Lam - (R - S + L) / mu
        Lam = (Lam + Lam.T) / 2
        residual = np.linalg.norm(R - S + L) / max(
            1, np.linalg.norm(R), np.linalg.norm(S), np.linalg.norm(L)
        )
        if residual < tolerance:
            break
        if iteration % 6 == 0:
            mu = max(mu / 4, 0.01)
    return S, L


def low_rank_factors(L: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """H (p x rank) with H H' = L, from the positive eigenvalues of L."""
    w, U = np.linalg.eigh(L)
    keep = w > tol * max(1.0, float(w.max(initial=0)))
    return U[:, keep][:, ::-1] * np.sqrt(w[keep][::-1])


def sparsity(S: np.ndarray) -> float:
    """The fraction of zero off-diagonal entries of S."""
    off = ~np.eye(S.shape[0], dtype=bool)
    return float(np.mean(S[off] == 0))
