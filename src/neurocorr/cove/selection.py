"""
Selection of active cells for the noise-correlation analysis.

Python port of ``+covest/ActiveCells.m`` in https://github.com/atlab/cov-est (MIT License,
(c) 2014 Tolias Lab). The criterion uses the summed activity in each quarter of the
recording, as the code does; the paper describes it in terms of variance.
"""

from __future__ import annotations

import numpy as np


def active_cells(X: np.ndarray, evoked_bins: int) -> np.ndarray:
    """
    Select the cells that stay active throughout the recording.

    The evoked bins of every trial are split into four quarters in trial order. A cell
    is selected if its total activity in every quarter exceeds 1% of the median quarter
    total across cells, and 1% of its own most active quarter.

    X : array (bins, conditions, trials, cells). Returns a boolean array (cells,).
    """
    p = X.shape[3]
    # rows ordered bin, then condition, then trial, as MATLAB's reshape(X, [], p)
    Y = np.asarray(X[:evoked_bins], dtype=np.float64).reshape(-1, p, order="F")
    Y = Y[~np.isnan(Y).any(axis=1)]
    # quarter of each row: MATLAB round(linspace(0.51, 4.49, n)), rounding halves up
    quarter = np.floor(np.linspace(0.51, 4.49, Y.shape[0]) + 0.5).astype(int) - 1
    g = np.zeros((4, p))
    np.add.at(g, quarter, Y)
    return np.all(g > 0.01 * np.median(g), axis=0) & np.all(
        g > 0.01 * g.max(axis=0), axis=0
    )
