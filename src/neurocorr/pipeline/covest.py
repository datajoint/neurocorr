"""
Noise-correlation estimation schema.

A translation of the DataJoint MATLAB schema ``covest`` in https://github.com/atlab/cov-est
(MIT License, (c) 2014 Tolias Lab), with each table named for what one row is:

    Session ──> Site ··> TrialTraceSet ──> CellSelection ──┐
                               │                    Method ├──> CovMatrix
                               └──> OriTuning         Fold ┘

The MATLAB schema's ``Traces`` (here ``TrialTraceSet``) was copied from an upstream lab pipeline
and keyed by its identifiers there. Here a session is numbered, a site is numbered within its
session, and a trial trace set has a simple number; each keeps its original identifiers as
attributes. The dotted arrow is a secondary reference: a trace set belongs to a site without
being identified by it. The ``ingest`` schema fills Session, Site, and TrialTraceSet from the
published data files.
"""

import os
import platform
import time

import datajoint as dj
import numpy as np

from neurocorr import cove

schema = dj.Schema(os.environ.get("DATABASE_PREFIX", "") + "covest")


@schema
class Session(dj.Entry):
    definition = """
    # An imaging session: one mouse on the setup
    session            : int16
    ---
    setup              : int16
    session_start_time : int64      # the session's identifier in the original database
    unique index (session_start_time)
    """


@schema
class Site(dj.Entry):
    definition = """
    # An imaging site: one 3D scan of a session, whose cells are recorded in each of its stimulus sets
    -> Session
    site                : int16     # numbered within the session
    ---
    aod_scan_start_time : int64     # the scan's identifier in the original database
    unique index (aod_scan_start_time)
    """


@schema
class TrialTraceSet(dj.Entry):
    definition = """
    # The binned activity of one imaging site in every trial of one stimulus set, aligned on stimulus onset
    trace_set             : int16
    ---
    -> Site
    high_repeats          : bool       # true: 2 or 5 directions with many trials, for correlations; false: 16 directions, for tuning
    preprocess_method_num : int32      # 5: spike inference by fast nonnegative deconvolution
    latency_ms            : int16      # presumed screen-to-V1 latency
    bin_ms                : int16      # bin duration
    evoked_bins           : int16      # bins driven by the stimulus; the remaining bins are spontaneous
    nneurons              : int16
    cell_xyz              : <npy@>     # nneurons x 3: cell centers, µm from the center of the scan
    cellnums              : <npy@>     # nneurons: cell numbers in the original segmentation
    ndirs                 : int16      # number of grating directions
    directions            : <npy@>     # ndirs: degrees
    ntrials               : <npy@>     # ndirs: trials of each direction
    trace_segments        : <npy@>     # bins x ndirs x trials x nneurons, float32, NaN-padded to the most trials
    unique index (session, site, high_repeats)
    """


def scan_id(key) -> int:
    """The trace set's scan identifier in the original database: the seed of its random draws."""
    return int((Site * TrialTraceSet & key).fetch1("aod_scan_start_time"))


@schema
class CellSelection(dj.Compute):
    definition = """
    # The cells of a trial trace set that stay active throughout the recording, for the noise-correlation analysis
    -> TrialTraceSet
    ---
    ncells    : int16
    selection : <npy@>      # nneurons booleans
    """

    @property
    def key_source(self):
        return super().key_source & (TrialTraceSet & {"high_repeats": True}).proj()

    def make(self, key):
        """Keep the cells active in every quarter of the recording.

        Over the evoked bins, in trial order, a cell's total activity in each quarter of the
        trials must exceed 1% of the median quarter total across cells and 1% of its own most
        active quarter (``+covest/ActiveCells.m``).
        """
        X, evoked_bins = (TrialTraceSet & key).fetch1("trace_segments", "evoked_bins")
        selection = cove.active_cells(np.asarray(X), evoked_bins)
        self.insert1({**key, "ncells": int(selection.sum()), "selection": selection})


@schema
class OriTuning(dj.Compute):
    definition = """
    # The von Mises orientation tuning of every cell of a 16-direction trial trace set, with its significance by shuffling
    -> TrialTraceSet
    ---
    nshuffles   : int16
    von_p_value : <npy@>      # nneurons: shuffle p-values
    von_r2      : <npy@>      # nneurons: R-squared of the fit against single-trial responses
    von_pref    : <npy@>      # nneurons: preferred direction, radians in [0, 2 pi)
    von_base    : <npy@>      # nneurons: base
    von_amp1    : <npy@>      # nneurons: amplitude at the preferred direction
    von_amp2    : <npy@>      # nneurons: amplitude at the opposite direction
    von_sharp   : <npy@>      # nneurons: sharpness
    """

    n_shuffles = 10000

    @property
    def key_source(self):
        return super().key_source & (TrialTraceSet & "ndirs >= 8").proj()

    def make(self, key):
        """Fit each cell's mean response over the evoked bins and test its tuning by shuffling.

        The response of a trial is the sum of its evoked bins. A two-peaked von Mises curve is
        fit to each cell's mean response per direction; R-squared is computed against single
        trials, and its p-value from 10,000 shuffles of the direction-trial labels, the same for
        all cells (``+covest/OriTuning.m``, ``+cove/VonMises2.m``).
        """
        X, evoked_bins = (TrialTraceSet & key).fetch1("trace_segments", "evoked_bins")
        X = np.asarray(X, dtype=np.float64)
        responses = (
            X[: min(evoked_bins, X.shape[0])].sum(axis=0).transpose(2, 0, 1)
        )  # cells x dirs x trials
        rng = np.random.default_rng([scan_id(key), self.n_shuffles])
        w, r2, p = cove.tuning_significance(responses, self.n_shuffles, rng)
        self.insert1(
            {
                **key,
                "nshuffles": self.n_shuffles,
                "von_p_value": p.astype(np.float32),
                "von_r2": r2.astype(np.float32),
                "von_pref": w[:, 4].astype(np.float32),
                "von_base": w[:, 0].astype(np.float32),
                "von_amp1": w[:, 1].astype(np.float32),
                "von_amp2": w[:, 2].astype(np.float32),
                "von_sharp": w[:, 3].astype(np.float32),
            }
        )


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
        {
            "method": 30,
            "include_spont": True,
            "condition_idx": 0,
            "regularization": "factor",
            "hyperparam_space": [
                {"name": "variance_shrinkage", "log": [-6, -1, 0.05]},
                {"name": "n_factors", "range": [0, 70]},
            ],
        },
        {
            "method": 80,
            "include_spont": True,
            "condition_idx": 0,
            "regularization": "glasso",
            "hyperparam_space": [{"name": "alpha", "log": [-5, -1, 0.05]}],
        },
        {
            "method": 90,
            "include_spont": True,
            "condition_idx": 0,
            "regularization": "lv-glasso",
            "hyperparam_space": [
                {"name": "alpha", "log": [-7, -1, 0.05]},
                {"name": "beta", "log": [-5, -1, 0.05]},
            ],
        },
        {
            "method": 91,
            "include_spont": True,
            "condition_idx": 0,
            "regularization": "lv-glasso refit",
            "hyperparam_space": [
                {"name": "alpha", "log": [-7, -1, 0.05]},
                {"name": "max_latent", "range": [0, 100]},
            ],
        },
        {
            "method": 100,
            "include_spont": True,
            "condition_idx": 0,
            "regularization": "lv-glasso",
            "hyperparam_space": [
                {"name": "alpha", "log": [-7, -1.5, 0.05]},
                {"name": "beta", "log": [-5, -1.5, 0.05]},
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
    -> CellSelection
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
        eligible = (CellSelection & "ncells > 100") * TrialTraceSet.proj(
            "ndirs"
        ) * Method.proj("condition_idx") & "ndirs = 2 OR condition_idx = 0"
        return (
            super().key_source
            & (TrialTraceSet & {"preprocess_method_num": 5}).proj()
            & eligible.proj()
        )

    def make(self, key):
        """Estimate the noise correlations of the active cells on the fold's training trials.

        Hyperparameters are chosen by 10-fold cross-validation within the training trials: a
        random search on the method's grids seeds a pattern search, each point scored by the
        normal validation loss with the variance regularization delta chosen per inner fold.
        The estimate is then refit on all training trials and scored on the fold's test trials
        (``cv_loss``). With ``nfolds = 1`` the training trials are all the trials and there is
        no test loss (``+covest/CovMatrix.m``).
        """
        start = time.time()
        X, evoked_bins = (TrialTraceSet & key).fetch1("trace_segments", "evoked_bins")
        selection = np.asarray((CellSelection & key).fetch1("selection"), dtype=bool)
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
                [scan_id(key), key["method"], key["k"], key["nfolds"]]
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
