"""
Covariance estimation for neural population activity.

Python port of the ``+cove`` package of https://github.com/atlab/cov-est
(MIT License, (c) 2014 Tolias Lab), the code of Yatsenko et al. (2015),
"Improved estimation and interpretation of correlations in neural circuits",
PLOS Comput Biol 11(3):e1004083.
"""

from neurocorr.cove.crossval import (
    DELTAS,
    cross_estimate_hyper,
    find_best_delta,
    fold_assignment,
    grid,
    split_trials,
    vloss,
)
from neurocorr.cove.estimate import condition_moments, corrcov, estimate, nan_cov
from neurocorr.cove.selection import active_cells

__all__ = [
    "DELTAS",
    "active_cells",
    "condition_moments",
    "corrcov",
    "cross_estimate_hyper",
    "estimate",
    "find_best_delta",
    "fold_assignment",
    "grid",
    "nan_cov",
    "split_trials",
    "vloss",
]
