"""
Cross-validation of covariance estimators: trial splits, the validation loss, and the
hyperparameter search.

Python port of ``+cove/splitTrials.m``, ``vloss.m``, ``findBestDelta.m`` and
``crossEstimateHyper.m`` in https://github.com/atlab/cov-est (MIT License,
(c) 2014 Tolias Lab). Differences from the MATLAB code:

- Trial slots are assigned to folds by a numpy permutation (seed 0) instead of MATLAB's
  ``rng(0); randperm``, so fold membership differs while keeping the same structure: every
  trial slot goes to one fold for all conditions, and all methods see the same folds.
- ``find_best_delta`` returns the grid value with the lowest loss. ``findBestDelta.m``
  returns the value one grid step past it.
- The random phase of the hyperparameter search takes a seeded generator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from neurocorr.cove.estimate import estimate, nan_cov

# variance regularization toward each cell's mean variance across conditions
DELTAS = np.exp(np.linspace(-4, 0, 81))
INNER_FOLDS = 10


def fold_assignment(n_trials: int, n_folds: int) -> np.ndarray:
    """The fold (1..n_folds) of each trial slot: equal counts, in a fixed random order."""
    folds = np.arange(1, n_trials + 1) % n_folds + 1
    return np.random.default_rng(0).permutation(folds)


def split_trials(
    X: np.ndarray, fold: int, n_folds: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split the trials of X into training data and the test data of one fold."""
    if n_folds == 1:
        return X, None
    folds = fold_assignment(X.shape[2], n_folds)
    return X[:, :, folds != fold], X[:, :, folds == fold]


def log_det(R: np.ndarray) -> float:
    """log det R, or NaN if R has a negative eigenvalue."""
    eigenvalues = np.linalg.eigvalsh(R)
    return float(np.sum(np.log(eigenvalues))) if np.all(eigenvalues >= 0) else math.nan


def vloss(
    X_test: np.ndarray, R: np.ndarray, M: np.ndarray, V: np.ndarray, delta: float
) -> float:
    """
    Validation loss: the normal (Gaussian) loss of the estimate on the test trials.

    The test data are centered with the training means M and scaled by the training
    variances V, shrunk by ``delta`` toward each cell's mean variance across conditions:
    L = (tr(R_test R^-1) + log det R) / p + sum over (bin, condition, cell) of the
    trial-weighted log V. Returns inf when the loss is undefined.
    """
    p = X_test.shape[3]
    N = np.sum(~np.isnan(X_test), axis=2, keepdims=True)
    N = N / N.sum()
    if delta:
        V = (1 - delta) * V + delta * np.nanmean(V.reshape(-1, p), axis=0)
    R_test = nan_cov(((X_test - M) / np.sqrt(V)).reshape(-1, p))
    try:
        trace = np.trace(np.linalg.solve(R, R_test))
    except np.linalg.LinAlgError:
        return math.inf
    loss = (trace + log_det(R)) / p + np.sum(np.log(V) * N)
    return math.inf if math.isnan(loss) else float(loss)


def find_best_delta(
    X_test: np.ndarray, R: np.ndarray, M: np.ndarray, V: np.ndarray
) -> float:
    """The variance regularization in DELTAS with the lowest loss, by a walk downhill from the middle."""
    losses = np.full(len(DELTAS), math.nan)
    i1 = len(DELTAS) // 2  # the middle pair, as in findBestDelta.m
    i2 = i1 + 1
    losses[i1] = vloss(X_test, R, M, V, DELTAS[i1])
    losses[i2] = vloss(X_test, R, M, V, DELTAS[i2])
    walk = (
        range(i1 - 1, -1, -1) if losses[i1] < losses[i2] else range(i2 + 1, len(DELTAS))
    )
    for i in walk:
        losses[i] = vloss(X_test, R, M, V, DELTAS[i])
        if losses[i] > np.nanmin(losses):
            break
    return float(DELTAS[np.nanargmin(losses)])


def grid(spec: dict) -> np.ndarray:
    """The values of one hyperparameter grid: {"log": [start, stop, step]} or {"range": [start, stop]}."""
    if "log" in spec:
        start, stop, step = spec["log"]
        return np.exp(start + step * np.arange(round((stop - start) / step) + 1))
    start, stop = spec["range"]
    return np.arange(start, stop + 1, dtype=np.float64)


def cross_estimate_hyper(
    X: np.ndarray,
    evoked_bins: int,
    regularization: str,
    space: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Choose hyperparameters by K-fold cross-validation on X.

    A random search over the grids seeds a pattern search that visits the diagonal
    neighbors of the best point at a halving step. Each visit estimates on the training
    trials of every inner fold and averages the validation loss on the test trials, with
    the variance regularization delta chosen per inner fold and averaged.

    Returns the hyperparameter values, their delta, the visited grid indices
    (visits x hyperparameters, 0-based), and the loss of each visit.
    """
    dims = np.array([len(g) for g in space])
    n_hypers = len(dims)
    pattern = dims > 1
    visited: dict[tuple, int] = {}
    order: list[tuple] = []
    losses: list[float] = []
    best_delta = math.nan
    inner = [split_trials(X, k, INNER_FOLDS) for k in range(1, INNER_FOLDS + 1)]
    one_condition = X.shape[0] == 1 and X.shape[1] == 1

    def visit(ix) -> None:
        nonlocal best_delta
        ix = tuple(int(i) for i in np.clip(ix, 0, dims - 1))
        if ix in visited:
            return
        hypers = [g[i] for g, i in zip(space, ix)]
        fits = [
            (test, *estimate(train, evoked_bins, regularization, hypers)[:3])
            for train, test in inner
        ]
        delta = (
            0.0
            if one_condition
            else float(np.mean([find_best_delta(*fit) for fit in fits]))
        )
        loss = float(np.mean([vloss(*fit, delta) for fit in fits]))
        visited[ix] = len(order)
        order.append(ix)
        losses.append(loss)
        if loss == min(losses):
            best_delta = delta

    # coarse random search to seed the local search
    decimate = 6
    n_random = math.ceil(np.prod(np.maximum(1, dims / decimate) / pattern.sum()))
    for _ in range(n_random):
        visit([rng.integers(d) for d in dims])

    # pattern search
    step = np.minimum(dims // 2, np.ceil(dims / decimate * 2)).astype(int)
    best_loss = min(losses)
    while True:
        last_best = best_loss
        j = int(np.argmin(losses))
        best_loss = losses[j]
        if best_loss == last_best and np.all(step <= 1):
            break
        step = np.ceil(step / 2).astype(int)
        for b in range(2 ** int(pattern.sum())):
            signs = np.ones(n_hypers, dtype=int)
            signs[pattern] = [
                2 * ((b >> bit) & 1) - 1 for bit in range(int(pattern.sum()))
            ]
            visit(np.array(order[j]) + step * signs)

    j = int(np.argmin(losses))
    hypers = np.array([g[i] for g, i in zip(space, order[j])])
    return hypers, best_delta, np.array(order), np.array(losses)
