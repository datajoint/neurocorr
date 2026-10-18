"""
Noise-correlation estimation schema.

A translation of the DataJoint MATLAB schema ``covest`` in https://github.com/atlab/cov-est
(MIT License, (c) 2014 Tolias Lab):

    SourceFile ──> SourceImport ··· fills ···> Traces ──> ActiveCells ──┐
                                                                 Method ├──> CovMatrix
                                                                   Fold ┘

The MATLAB schema's ``Traces`` was filled from an upstream lab pipeline. Here, the published
data files are its source: ``SourceImport`` reads each file and inserts its recordings into
``Traces`` (fan-out ingestion), which keeps the original table unchanged.
"""

import os
import platform
import time

import datajoint as dj
import numpy as np

from neurocorr import cove
from neurocorr.ingest import SOURCE_FILES, fetch, read_records, to_row

schema = dj.Schema(os.environ.get("DATABASE_PREFIX", "") + "covest")


@schema
class SourceFile(dj.Lookup):
    definition = """
    # A data file of the original study, published with its code in github.com/atlab/cov-est
    source_file : varchar(24)
    ---
    sha256      : char(64)
    """
    contents = [
        {"source_file": name, "sha256": sha256} for name, sha256 in SOURCE_FILES
    ]


@schema
class SourceImport(dj.Ingest):
    definition = """
    # The import of a source file's recordings into Traces
    -> SourceFile
    ---
    n_records : int16
    """

    def make(self, key):
        path = fetch(key["source_file"], (SourceFile & key).fetch1("sha256"))
        rows = [to_row(record) for record in read_records(path)]
        Traces.insert(rows, skip_duplicates=True)
        self.insert1({**key, "n_records": len(rows)})


@schema
class Traces(dj.Entry):
    definition = """
    # Binned activity of a population in every trial of a stimulus set, aligned on stimulus onset
    subject_id            : int32
    setup                 : int16
    session_start_time    : int64      # the session's identifier in the original database
    aod_scan_start_time   : int64      # the scan's identifier in the original database: one imaging site
    preprocess_method_num : int32      # 5: spike inference by fast nonnegative deconvolution
    high_repeats          : bool       # true: 2 or 5 directions with many trials, for correlations; false: 16 directions, for tuning
    ---
    latency_ms            : int16      # presumed screen-to-V1 latency
    bin_ms                : int16      # bin duration
    evoked_bins           : int16      # bins driven by the stimulus; the remaining bins are spontaneous
    nneurons              : int16
    cell_xyz              : <npy@>      # nneurons x 3: cell centers, µm from the center of the scan
    cellnums              : <npy@>      # nneurons: cell numbers in the original segmentation
    ndirs                 : int16      # number of grating directions
    directions            : <npy@>      # ndirs: degrees
    ntrials               : <npy@>      # ndirs: trials of each direction
    trace_segments        : <npy@>      # bins x ndirs x trials x nneurons, float32, NaN-padded to the most trials
    """


@schema
class ActiveCells(dj.Compute):
    definition = """
    # Cells active throughout the recording, selected for the noise-correlation analysis
    -> Traces
    ---
    ncells    : int16
    selection : <npy@>       # nneurons booleans
    """

    @property
    def key_source(self):
        return super().key_source & {"high_repeats": True}

    def make(self, key):
        X, evoked_bins = (Traces & key).fetch1("trace_segments", "evoked_bins")
        selection = cove.active_cells(np.asarray(X), evoked_bins)
        self.insert1({**key, "ncells": int(selection.sum()), "selection": selection})


@schema
class Method(dj.Lookup):
    definition = """
    # A covariance estimator and the grid of its hyperparameters
    method           : int16
    ---
    include_spont    : bool       # include the spontaneous bins
    condition_idx    : int16      # 0: all directions; 1 or 2: that direction only
    regularization   : enum('sample', 'diag', 'factor', 'glasso', 'lv-glasso', 'lv-glasso refit')
    hyperparam_space : json       # one grid per hyperparameter, in the order estimate() takes them
    """
    contents = [
        {
            "method": 0,
            "include_spont": True,
            "condition_idx": 0,
            "regularization": "sample",
            "hyperparam_space": [],
        },
        {
            "method": 1,
            "include_spont": False,
            "condition_idx": 0,
            "regularization": "sample",
            "hyperparam_space": [],
        },
        {
            "method": 2,
            "include_spont": False,
            "condition_idx": 1,
            "regularization": "sample",
            "hyperparam_space": [],
        },
        {
            "method": 3,
            "include_spont": False,
            "condition_idx": 2,
            "regularization": "sample",
            "hyperparam_space": [],
        },
        {
            "method": 10,
            "include_spont": True,
            "condition_idx": 0,
            "regularization": "diag",
            "hyperparam_space": [
                {"name": "variance_shrinkage", "log": [-6, -1, 0.05]},
                {"name": "correlation_shrinkage", "log": [-4, 0, 0.05]},
            ],
        },
    ]


@schema
class Fold(dj.Lookup):
    definition = """
    # A fold of K-fold cross-validation; nfolds = 1 estimates from all the trials
    k      : int16
    nfolds : int16
    """
    contents = [{"k": 1, "nfolds": 1}] + [{"k": k, "nfolds": 10} for k in range(1, 11)]


@schema
class CovMatrix(dj.Compute):
    definition = """
    # Noise correlations estimated on the training trials of a fold, and their validation loss on its test trials
    -> ActiveCells
    -> Method
    -> Fold
    ---
    means           : <npy@>          # bins x conditions x 1 x ncells
    variances       : <npy@>          # bins x conditions x 1 x ncells
    corr_matrix     : <npy@>          # ncells x ncells correlation matrix
    sparse = null   : <npy@>          # sparse component of the precision matrix
    lowrank = null  : <npy@>          # low-rank component: loadings or latent factors
    hypers = null   : <npy@>          # the chosen hyperparameter values
    delta           : float64        # variance regularization toward each cell's mean variance
    visited = null  : <npy@>          # grid indices visited by the hyperparameter search (0-based)
    losses = null   : <npy@>          # mean inner validation loss at each visited index
    sparsity = 0    : float64        # fraction of zero off-diagonal entries in the sparse component
    cv_loss = null  : float64        # validation loss on the test trials of the fold (nfolds > 1)
    host            : varchar(255)
    computing_time  : float32        # seconds
    cm_ts = CURRENT_TIMESTAMP : datetime
    """

    @property
    def key_source(self):
        # the paper's sites; condition-specific methods only on sites with two directions
        eligible = (ActiveCells & "ncells > 100") * Traces.proj("ndirs") * Method.proj(
            "condition_idx"
        ) & "ndirs = 2 OR condition_idx = 0"
        return super().key_source & {"preprocess_method_num": 5} & eligible.proj()

    def make(self, key):
        start = time.time()
        X, evoked_bins = (Traces & key).fetch1("trace_segments", "evoked_bins")
        selection = np.asarray((ActiveCells & key).fetch1("selection"), dtype=bool)
        method = (Method & key).fetch1()

        X = np.asarray(X, dtype=np.float64)[..., selection]
        if not method["include_spont"]:
            X = X[:evoked_bins]
        if method["condition_idx"]:
            c = method["condition_idx"] - 1
            X = X[:, c : c + 1]

        X, X_test = cove.split_trials(X, key["k"], key["nfolds"])
        space = [cove.grid(spec) for spec in method["hyperparam_space"]]
        regularization = method["regularization"]
        visited = losses = None
        if any(len(g) > 1 for g in space):
            rng = np.random.default_rng(
                [key["aod_scan_start_time"], key["method"], key["k"], key["nfolds"]]
            )
            hypers, delta, visited, losses = cove.cross_estimate_hyper(
                X, evoked_bins, regularization, space, rng
            )
        else:
            hypers = np.array([g[0] for g in space])
            inner = [
                cove.split_trials(X, k, cove.crossval.INNER_FOLDS)
                for k in range(1, cove.crossval.INNER_FOLDS + 1)
            ]
            delta = float(
                np.mean(
                    [
                        cove.find_best_delta(
                            test,
                            *cove.estimate(train, evoked_bins, regularization, hypers)[
                                :3
                            ],
                        )
                        for train, test in inner
                    ]
                )
            )

        R, M, V, extras = cove.estimate(X, evoked_bins, regularization, hypers)
        cv_loss = None if X_test is None else cove.vloss(X_test, R, M, V, delta)
        if cv_loss is not None and not np.isfinite(cv_loss):
            cv_loss = None  # an undefined loss; vloss returns inf
        self.insert1(
            {
                **key,
                "means": M.astype(np.float32),
                "variances": V.astype(np.float32),
                "corr_matrix": R,
                "sparse": extras.get("sparse"),
                "lowrank": extras.get("lowrank"),
                "hypers": hypers if len(hypers) else None,
                "delta": delta,
                "visited": visited,
                "losses": losses,
                "sparsity": extras.get("sparsity", 0.0),
                "cv_loss": cv_loss,
                "host": platform.node(),
                "computing_time": time.time() - start,
            }
        )
