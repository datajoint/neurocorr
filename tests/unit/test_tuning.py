"""
Unit tests for the von Mises orientation tuning.
"""

import numpy as np

from neurocorr import cove


def curves(n=40, seed=0):
    rng = np.random.default_rng(seed)
    w = np.column_stack(
        [
            rng.uniform(1, 3, n),  # base
            rng.uniform(1, 2, n),  # preferred amplitude
            rng.uniform(0, 0.8, n),  # opposite amplitude, smaller
            rng.uniform(1.5, 6, n),  # sharpness
            rng.uniform(0, 2 * np.pi, n),  # preferred direction
        ]
    )
    return w


def angle_error(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def test_fit_recovers_noise_free_curves():
    w = curves()
    fitted = cove.fit_von_mises(cove.von_mises(w, 16))
    np.testing.assert_allclose(
        cove.von_mises(fitted, 16), cove.von_mises(w, 16), atol=0.02
    )
    assert np.median(angle_error(fitted[:, 4], w[:, 4])) < 0.05
    assert np.all(fitted[:, 1] >= fitted[:, 2]) and np.all(fitted[:, 2] >= 0)


def test_the_larger_peak_is_preferred():
    w = curves(n=1)
    w[0, 1], w[0, 2] = 0.3, 1.5  # the opposite peak is larger
    fitted = cove.fit_von_mises(cove.von_mises(w, 16))
    assert angle_error(fitted[0, 4], w[0, 4] + np.pi) < 0.05
    assert fitted[0, 1] > fitted[0, 2]


def test_shuffle_test_separates_tuned_from_untuned_cells():
    rng = np.random.default_rng(1)
    w = curves(n=20)
    w[10:, 1:3] = 0  # cells 10-19 are untuned
    mean = cove.von_mises(w, 16)
    X = rng.poisson(np.repeat(mean[:, :, None], 20, axis=2) * 2) / 2.0
    X[:, 3, -1] = np.nan  # one direction has a trial fewer
    fitted, r2, p = cove.tuning_significance(X, 400, np.random.default_rng(2))
    assert fitted.shape == (20, 5) and r2.shape == p.shape == (20,)
    assert np.all(p[:10] < 0.05)
    assert np.mean(p[10:] < 0.05) <= 0.2
    again = cove.tuning_significance(X, 400, np.random.default_rng(2))[2]
    np.testing.assert_array_equal(p, again)
