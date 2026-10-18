"""
Covariance estimation from trial-aligned population activity.

Python port of ``+cove/estimate.m`` in https://github.com/atlab/cov-est (MIT License,
(c) 2014 Tolias Lab), the code of Yatsenko et al. (2015), PLOS Comput Biol 11(3):e1004083.

Data arrays have the shape (bins, conditions, trials, cells), with NaN for the missing
trials of conditions that have fewer trials than others.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def corrcov(C: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def nan_cov(Y: np.ndarray) -> np.ndarray:
    """Second moment Y'Y/n over the rows of Y with no NaN, without subtracting a mean."""
    Y = Y[~np.isnan(Y).any(axis=1)]
    return Y.T @ Y / Y.shape[0]


def condition_moments(X: np.ndarray, evoked_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Means and variances of each cell's response, per bin and condition.

    Evoked bins get their own mean and variance over trials, per bin and condition.
    The remaining (spontaneous) bins share one mean and one variance per cell across
    bins, conditions, and trials. Variances are normalized by n.

    Returns M and V, each of shape (bins, conditions, 1, cells).
    """
    n_bins, n_conds, _, p = X.shape
    E = min(n_bins, evoked_bins)
    M = np.nanmean(X[:E], axis=2, keepdims=True)
    if E < n_bins:
        m = np.nanmean(X[E:].reshape(-1, p), axis=0)
        M = np.concatenate([M, np.broadcast_to(m, (n_bins - E, n_conds, 1, p))])
    Xc = X - M
    V = np.nanvar(Xc[:E], axis=2, keepdims=True)
    if E < n_bins:
        v = np.nanvar(Xc[E:].reshape(-1, p), axis=0)
        V = np.concatenate([V, np.broadcast_to(v, (n_bins - E, n_conds, 1, p))])
    return M, V


def estimate(
    X: np.ndarray, evoked_bins: int, regularization: str, hypers: Sequence[float] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Estimate the noise correlation matrix of a population.

    Parameters
    ----------
    X : array (bins, conditions, trials, cells)
    evoked_bins : the number of bins driven by the stimulus; later bins are spontaneous
    regularization : 'sample' or 'diag'
    hypers : the hyperparameter values, in the order of the method's grids

    Returns
    -------
    R : (cells, cells) correlation matrix
    M : (bins, conditions, 1, cells) mean responses
    V : (bins, conditions, 1, cells) variances; for a regularized estimate, scaled by the
        change in each cell's variance that the regularization made
    extras : additional parts of the estimate (empty for 'sample' and 'diag')
    """
    X = np.asarray(X, dtype=np.float64)
    p = X.shape[3]
    M, V = condition_moments(X, evoked_bins)

    # the sample correlation of z-scores, pooled over bins, conditions, and trials
    R = corrcov(nan_cov(((X - M) / np.sqrt(V)).reshape(-1, p)))

    # the sample covariance at the average variances
    sigma = np.sqrt(V.reshape(-1, p).mean(axis=0))
    C = R * np.outer(sigma, sigma)
    extras: dict = {}

    if regularization == "sample":
        if len(hypers):
            raise ValueError("the sample estimator takes no hyperparameters")
    elif regularization == "diag":
        # shrink the variances toward their median, and the correlations toward zero
        variance_shrinkage, correlation_shrinkage = hypers
        v = np.diag(C)
        s = np.sqrt((1 - variance_shrinkage) * v + variance_shrinkage * np.median(v))
        r = (1 - correlation_shrinkage) * corrcov(C) + correlation_shrinkage * np.eye(p)
        C = r * np.outer(s, s)
    else:
        raise NotImplementedError(f"regularization {regularization!r}")

    R = C / np.outer(sigma, sigma)
    if regularization != "sample":
        # carry the regularized change in each cell's variance from R into V
        V = V * np.diag(R)
        R = corrcov(R)
    return R, M, V, extras
