"""
Unit tests for the factor, sparse, and sparse + latent estimators.
"""

import numpy as np
import pytest

from neurocorr import cove
from neurocorr.cove import structured


def factor_model(p=30, k=3, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 0.6, size=(p, k))
    return L @ L.T + np.diag(rng.uniform(0.5, 1.5, p))


def sparse_plus_latent(p=30, seed=0):
    """A covariance whose precision is a sparse chain minus a rank-2 component."""
    rng = np.random.default_rng(seed)
    S = np.eye(p) * 2.0
    for i in range(p - 1):
        S[i, i + 1] = S[i + 1, i] = -0.6
    H = rng.normal(0, 0.25, size=(p, 2))
    return np.linalg.inv(S - H @ H.T)


def sample_cov(Sigma, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(len(Sigma)), Sigma, size=n)
    return X.T @ X / n


def test_factor_analysis_recovers_a_factor_model():
    Sigma = factor_model()
    L, psi = structured.factor_analysis(sample_cov(Sigma, n=20000), 3)
    assert L.shape == (30, 3)
    np.testing.assert_allclose(L @ L.T + np.diag(psi), Sigma, atol=0.08)


def test_factor_analysis_without_factors_is_the_diagonal():
    C = sample_cov(factor_model())
    L, psi = structured.factor_analysis(C, 0)
    assert L.shape == (30, 0)
    np.testing.assert_allclose(psi, np.diag(C))


def test_glasso_with_a_small_penalty_approaches_the_inverse():
    C = sample_cov(sparse_plus_latent())
    S, L = structured.sparse_latent(C, 1e-4)
    assert not L.any()
    np.testing.assert_allclose(np.linalg.inv(S), C, rtol=0.05, atol=0.01)


def test_glasso_sparsity_grows_with_the_penalty():
    C = sample_cov(sparse_plus_latent())
    weak = structured.sparsity(structured.sparse_latent(C, 0.01)[0])
    strong = structured.sparsity(structured.sparse_latent(C, 0.1)[0])
    assert strong > weak


def test_sparse_latent_has_a_positive_semidefinite_low_rank_part():
    C = sample_cov(sparse_plus_latent())
    S, L = structured.sparse_latent(C, 0.05, 0.05)
    assert np.linalg.eigvalsh(L).min() > -1e-8
    H = structured.low_rank_factors(L)
    np.testing.assert_allclose(H @ H.T, L, atol=1e-8)
    assert np.linalg.eigvalsh(S - L).min() > 0


def test_capped_rank():
    C = sample_cov(sparse_plus_latent())
    _, L = structured.sparse_latent_capped(C, 0.05, max_rank=1)
    assert np.sum(np.linalg.eigvalsh(L) > 1e-8) <= 1


@pytest.mark.parametrize(
    "regularization, hypers",
    [
        ("factor", [0.1, 3]),
        ("glasso", [0.05]),
        ("lv-glasso", [0.05, 0.1]),
        ("lv-glasso refit", [0.05, 2]),
    ],
)
def test_estimate_returns_a_correlation_matrix(regularization, hypers):
    rng = np.random.default_rng(0)
    p = 20
    noise = rng.multivariate_normal(
        np.zeros(p), sparse_plus_latent(p), size=(3, 2, 200)
    )
    X = rng.uniform(1, 3, size=(3, 2, 1, p)) + noise
    R, _, _, extras = cove.estimate(X, 2, regularization, hypers)
    np.testing.assert_allclose(np.diag(R), 1)
    assert np.linalg.eigvalsh(R).min() > 0
    if regularization != "factor":
        assert extras["sparse"].shape == (p, p) and 0 <= extras["sparsity"] <= 1
