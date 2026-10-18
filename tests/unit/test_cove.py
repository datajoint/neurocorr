"""
Unit tests for the covariance estimators and cross-validation.
"""

import numpy as np
import pytest

from neurocorr import cove


def simulate(n_bins=3, n_conds=2, n_trials=60, p=12, seed=0, rho=0.3):
    """Gaussian trials with a shared correlation and condition-specific means."""
    rng = np.random.default_rng(seed)
    true_corr = (1 - rho) * np.eye(p) + rho
    noise = rng.multivariate_normal(
        np.zeros(p), true_corr, size=(n_bins, n_conds, n_trials)
    )
    means = rng.uniform(1, 3, size=(n_bins, n_conds, 1, p))
    return means + noise, true_corr


class TestEstimate:
    def test_sample_is_a_correlation_matrix(self):
        X, _ = simulate()
        R, M, V, _ = cove.estimate(X, evoked_bins=2, regularization="sample")
        np.testing.assert_allclose(np.diag(R), 1)
        np.testing.assert_allclose(R, R.T)
        assert M.shape == V.shape == (3, 2, 1, 12)

    def test_sample_recovers_the_correlation(self):
        X, true_corr = simulate(n_trials=2000)
        R, *_ = cove.estimate(X, evoked_bins=3, regularization="sample")
        np.testing.assert_allclose(R, true_corr, atol=0.05)

    def test_spontaneous_bins_share_a_mean_and_variance(self):
        X, _ = simulate(n_bins=5)
        _, M, V, _ = cove.estimate(X, evoked_bins=2, regularization="sample")
        assert np.allclose(M[2:], M[2, 0]) and np.allclose(V[2:], V[2, 0])
        assert not np.allclose(M[0], M[1])

    def test_nan_padded_trials_are_ignored(self):
        X, _ = simulate()
        padded = np.concatenate([X, np.full((3, 2, 5, 12), np.nan)], axis=2)
        R1, *_ = cove.estimate(X, 2, "sample")
        R2, *_ = cove.estimate(padded, 2, "sample")
        np.testing.assert_allclose(R1, R2)

    def test_diag_without_shrinkage_is_the_sample(self):
        X, _ = simulate()
        R_sample, _, V_sample, _ = cove.estimate(X, 2, "sample")
        R_diag, _, V_diag, _ = cove.estimate(X, 2, "diag", [0.0, 0.0])
        np.testing.assert_allclose(R_diag, R_sample)
        np.testing.assert_allclose(V_diag, V_sample)

    def test_diag_full_correlation_shrinkage_is_the_identity(self):
        X, _ = simulate()
        R, *_ = cove.estimate(X, 2, "diag", [0.5, 1.0])
        np.testing.assert_allclose(R, np.eye(12), atol=1e-12)

    def test_diag_variance_shrinkage_moves_variances_toward_the_median(self):
        X, _ = simulate()
        X[..., 0] *= 3  # one cell with a large variance
        _, _, V0, _ = cove.estimate(X, 2, "sample")
        _, _, V1, _ = cove.estimate(X, 2, "diag", [0.5, 0.0])
        spread = lambda V: np.ptp(V.reshape(-1, 12).mean(axis=0))  # noqa: E731
        assert spread(V1) < spread(V0)

    def test_sample_rejects_hyperparameters(self):
        X, _ = simulate()
        with pytest.raises(ValueError):
            cove.estimate(X, 2, "sample", [0.1])


class TestSplit:
    def test_folds_are_balanced_and_fixed(self):
        folds = cove.fold_assignment(103, 10)
        counts = np.bincount(folds)[1:]
        assert counts.sum() == 103 and counts.max() - counts.min() <= 1
        np.testing.assert_array_equal(folds, cove.fold_assignment(103, 10))

    def test_split_partitions_the_trials(self):
        X, _ = simulate(n_trials=40)
        tests = []
        for k in range(1, 11):
            train, test = cove.split_trials(X, k, 10)
            assert train.shape[2] + test.shape[2] == 40
            tests.append(test)
        assert sum(t.shape[2] for t in tests) == 40

    def test_one_fold_has_no_test_trials(self):
        X, _ = simulate()
        train, test = cove.split_trials(X, 1, 1)
        assert train is X and test is None


class TestLoss:
    def test_true_correlation_beats_a_wrong_one(self):
        X, true_corr = simulate(n_trials=400, rho=0.4)
        train, test = cove.split_trials(X, 1, 10)
        _, M, V, _ = cove.estimate(train, 3, "sample")
        assert cove.vloss(test, true_corr, M, V, 0) < cove.vloss(
            test, np.eye(12), M, V, 0
        )

    def test_undefined_loss_is_infinite(self):
        X, _ = simulate()
        _, M, V, _ = cove.estimate(X, 2, "sample")
        indefinite = np.eye(12)
        indefinite[0, 1] = indefinite[1, 0] = 2.0
        assert cove.vloss(X, indefinite, M, V, 0) == np.inf

    def test_best_delta_has_the_lowest_loss_on_the_grid(self):
        X, _ = simulate(n_trials=200)
        train, test = cove.split_trials(X, 1, 10)
        R, M, V, _ = cove.estimate(train, 2, "sample")
        delta = cove.find_best_delta(test, R, M, V)
        assert delta in cove.DELTAS
        losses = [cove.vloss(test, R, M, V, d) for d in cove.DELTAS]
        i = int(np.argmin(losses))
        # the walk descends from the middle of the grid, so it finds the minimum of a unimodal loss
        assert delta == pytest.approx(cove.DELTAS[i])


class TestSearch:
    def test_grids(self):
        assert len(cove.grid({"log": [-6, -1, 0.05]})) == 101
        assert len(cove.grid({"log": [-4, 0, 0.05]})) == 81
        np.testing.assert_array_equal(cove.grid({"range": [0, 70]}), np.arange(71))
        assert cove.grid({"log": [-4, 0, 0.05]})[-1] == pytest.approx(1.0)

    def test_diag_search_returns_grid_values(self):
        X, _ = simulate(n_trials=100, p=8)
        space = [cove.grid({"log": [-6, -1, 0.25]}), cove.grid({"log": [-4, 0, 0.25]})]
        hypers, delta, visited, losses = cove.cross_estimate_hyper(
            X, 2, "diag", space, np.random.default_rng(0)
        )
        assert hypers[0] in space[0] and hypers[1] in space[1]
        assert visited.shape == (len(losses), 2)
        assert len({tuple(v) for v in visited}) == len(losses)
        assert losses.min() == losses[np.argmin(losses)] and np.isfinite(delta)

    def test_search_is_reproducible_with_a_seed(self):
        X, _ = simulate(n_trials=60, p=6)
        space = [cove.grid({"log": [-6, -1, 0.5]}), cove.grid({"log": [-4, 0, 0.5]})]
        a = cove.cross_estimate_hyper(X, 2, "diag", space, np.random.default_rng(7))
        b = cove.cross_estimate_hyper(X, 2, "diag", space, np.random.default_rng(7))
        np.testing.assert_array_equal(a[2], b[2])


class TestActiveCells:
    def test_silent_and_fading_cells_are_excluded(self):
        rng = np.random.default_rng(1)
        X = rng.exponential(1.0, size=(8, 2, 40, 5))
        X[..., 1] = 0  # silent
        X[:, :, 25:, 2] = 0  # silent in the last quarter of the 35 recorded trials
        X[:, :, 35:, :] = np.nan  # padded trials are dropped
        selection = cove.active_cells(X, evoked_bins=8)
        np.testing.assert_array_equal(selection, [True, False, False, True, True])
