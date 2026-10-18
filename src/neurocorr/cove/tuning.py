"""
Orientation tuning: two-peaked von Mises fits and their significance by shuffling.

Python port of ``+cove/VonMises2.m`` and ``+covest/OriTuning.m`` in
https://github.com/atlab/cov-est (MIT License, (c) 2014 Tolias Lab). The model of each
cell's mean response at direction phi is

    f(phi) = w1 + w2 exp(w4 (cos(phi - w5) - 1)) + w3 exp(w4 (cos(phi - w5 + pi) - 1)),

with w2 >= w3 >= 0: a base, the amplitudes at the preferred and the opposite direction, the
sharpness, and the preferred direction in [0, 2 pi). The directions are taken as uniformly
spaced from 0, as in the original.
"""

from __future__ import annotations

import numpy as np

MIN_SHARPNESS = 0.7


def _peaks(phi: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean peaks at the preferred direction and, shifted by half a turn, the opposite one."""
    g1 = np.exp(w[:, 3:4] * (np.cos(phi - w[:, 4:5]) - 1))
    g1 = g1 - g1.mean(axis=1, keepdims=True)
    return g1, np.roll(g1, len(phi) // 2, axis=1)


def _fit_linear(
    F: np.ndarray, phi: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The amplitudes for the sharpness and direction in w, clipped at zero, and the explained sum of squares."""
    g1, g2 = _peaks(phi, w)
    a = np.sum(
        g1 * g1, axis=1, keepdims=True
    )  # equals sum(g2 * g2): g2 is a shift of g1
    c = np.sum(g1 * g2, axis=1, keepdims=True)
    inverse = 1 / (a * a - c * c)
    h1 = inverse * (a * g1 - c * g2)
    h2 = inverse * (a * g2 - c * g1)
    w = w.copy()
    w[:, 1] = np.maximum(0, np.sum(h1 * F, axis=1))
    w[:, 2] = np.maximum(0, np.sum(h2 * F, axis=1))
    explained = np.sum((w[:, 1:2] * g1 + w[:, 2:3] * g2) ** 2, axis=1)
    return w, explained


def fit_von_mises(responses: np.ndarray) -> np.ndarray:
    """
    Fit a two-peaked von Mises curve to each row of ``responses`` (curves x directions).

    Returns w (curves x 5): base, preferred amplitude, opposite amplitude, sharpness, and
    preferred direction (radians).
    """
    F = np.asarray(responses, dtype=np.float64)
    n, k = F.shape
    if k % 2:
        raise ValueError("an even number of directions is required")
    M = F.mean(axis=1)
    F = F - M[:, None]
    phi = np.arange(k) / k * 2 * np.pi

    # initial solutions: seven preferred directions at a mid-level sharpness
    best = None
    for theta in np.arange(0, np.pi - 1e-5, np.pi / 7):
        trial = np.column_stack(
            [
                np.zeros(n),
                np.full(n, np.nan),
                np.full(n, np.nan),
                np.full(n, 5.0),
                np.full(n, theta),
            ]
        )
        trial, score = _fit_linear(F, phi, trial)
        if best is None:
            w, best = trial, score
        else:
            better = score > best
            w[better], best[better] = trial[better], score[better]

    theta_step, sharp_step = np.pi / 12, np.log(4)
    for iteration in range(1, 4):
        # preferred direction: a parabolic step between the two neighbors
        trial = w.copy()
        trial[:, 4] = w[:, 4] - theta_step
        _, left = _fit_linear(F, phi, trial)
        trial[:, 4] = w[:, 4] + theta_step
        _, right = _fit_linear(F, phi, trial)
        with np.errstate(divide="ignore", invalid="ignore"):
            adjust = (right - left) / (2 * best - left - right) / 2
        trial[:, 4] = w[:, 4] + theta_step * np.clip(np.nan_to_num(adjust), -1, 1)
        trial, score = _fit_linear(F, phi, trial)
        better = score >= best
        w[better], best[better] = trial[better], score[better]
        theta_step /= 10

        # sharpness: a bisection in log space above the minimum sharpness
        for _ in range(1 + iteration):
            for sign in (-1, 1):
                trial = w.copy()
                trial[:, 3] = (w[:, 3] - MIN_SHARPNESS) * np.exp(
                    sign * sharp_step
                ) + MIN_SHARPNESS
                trial, score = _fit_linear(F, phi, trial)
                better = score >= best
                w[better], best[better] = trial[better], score[better]
                sharp_step *= 0.75

    # the larger peak is the preferred direction
    flip = w[:, 1] < w[:, 2]
    w[flip, 1], w[flip, 2] = w[flip, 2], w[flip, 1].copy()
    w[flip, 4] += np.pi
    w[:, 4] %= 2 * np.pi

    # restore the base
    peak = np.exp(w[:, 3:4] * (np.cos(phi - w[:, 4:5]) - 1)).mean(axis=1)
    w[:, 0] = M - (w[:, 1] + w[:, 2]) * peak
    return w


def von_mises(w: np.ndarray, n_directions: int) -> np.ndarray:
    """The fitted curves (curves x directions) at uniformly spaced directions from 0."""
    phi = np.arange(n_directions) / n_directions * 2 * np.pi
    c = np.cos(phi - w[:, 4:5])
    return (
        w[:, 0:1]
        + w[:, 1:2] * np.exp(w[:, 3:4] * (c - 1))
        + w[:, 2:3] * np.exp(w[:, 3:4] * (-c - 1))
    )


def _r2(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """R² of the fitted curves against single-trial responses X (curves x directions x trials)."""
    residual = X - von_mises(w, X.shape[1])[:, :, None]
    flat = X.reshape(len(X), -1)
    return 1 - np.nanvar(residual.reshape(len(X), -1), axis=1) / np.nanvar(flat, axis=1)


def tuning_significance(
    X: np.ndarray, n_shuffles: int, rng: np.random.Generator, batch: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Von Mises tuning of every cell and its significance.

    X : responses, cells x directions x trials, with NaN for missing trials. The fit uses
    the mean over trials, and R² is computed against single trials. For each shuffle, the
    direction-trial labels are permuted jointly for all cells and the fit is repeated;
    p = (0.5 + the number of shuffles with R² at least the observed one) / n_shuffles.

    Returns w (cells x 5), r2 (cells,), and p (cells,).
    """
    X = np.asarray(X, dtype=np.float64)
    cells, n_dirs, n_trials = X.shape
    w = fit_von_mises(np.nanmean(X, axis=2))
    r2 = _r2(X, w)
    # MATLAB's reshape order: direction fastest, then trial
    flat = X.transpose(0, 2, 1).reshape(cells, -1)
    count = np.zeros(cells)
    for start in range(0, n_shuffles, batch):
        size = min(batch, n_shuffles - start)
        order = np.argsort(rng.random((size, n_dirs * n_trials)), axis=1)
        shuffled = (
            flat[:, order]
            .transpose(1, 0, 2)
            .reshape(size * cells, n_trials, n_dirs)
            .transpose(0, 2, 1)
        )
        ws = fit_von_mises(np.nanmean(shuffled, axis=2))
        count += (_r2(shuffled, ws).reshape(size, cells) >= r2).sum(axis=0)
    return w, r2, (0.5 + count) / n_shuffles
