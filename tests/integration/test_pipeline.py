"""
Integration tests: the pipeline on a database, with simulated recordings.
"""

import numpy as np
import pytest


def simulated_traces(n_cells=110, n_dirs=2, n_trials=30, high_repeats=True, scan=1):
    rng = np.random.default_rng(scan)
    n_bins = 4 if high_repeats else 3
    traces = rng.exponential(size=(n_bins, n_dirs, n_trials, n_cells)).astype(
        np.float32
    )
    traces[:, -1, -1] = np.nan
    return {
        "subject_id": 16,
        "setup": 4,
        "session_start_time": 1000,
        "aod_scan_start_time": 1000 + scan,
        "preprocess_method_num": 5,
        "high_repeats": high_repeats,
        "latency_ms": 20,
        "bin_ms": 150,
        "evoked_bins": 2 if high_repeats else 4,
        "nneurons": n_cells,
        "cell_xyz": rng.uniform(-99, 99, size=(n_cells, 3)),
        "cellnums": np.arange(1, n_cells + 1),
        "ndirs": n_dirs,
        "directions": np.linspace(0, 90, n_dirs),
        "ntrials": np.array([n_trials] * (n_dirs - 1) + [n_trials - 1]),
        "trace_segments": traces,
    }


@pytest.mark.integration
def test_source_import_fills_traces(clean_schema, tmp_path, monkeypatch):
    from scipy.io import savemat

    covest = clean_schema
    rows = [
        simulated_traces(scan=1),
        simulated_traces(scan=2, high_repeats=False, n_dirs=16),
    ]
    fields = [
        "subject_id",
        "setup",
        "session_start_time",
        "aod_scan_start_time",
        "preprocess_method_num",
        "high_repeats",
        "latency_ms",
        "bin_ms",
        "evoked_bins",
        "nneurons",
        "cell_xyz",
        "cellnums",
        "ndirs",
        "directions",
        "ntrials",
        "trace_segments",
    ]
    tuples = np.zeros((2,), dtype=[(f, object) for f in fields])
    for i, row in enumerate(rows):
        tuples[i] = tuple(
            int(row[f]) if f == "high_repeats" else row[f] for f in fields
        )
    path = tmp_path / "traces.mat"
    savemat(path, {"tuples": tuples})
    monkeypatch.setattr(covest, "fetch", lambda file_name, sha256: path)

    assert len(covest.SourceFile()) == 4
    covest.SourceImport.populate({"source_file": "traces-pre5-0000.mat"})
    assert (covest.SourceImport & {"source_file": "traces-pre5-0000.mat"}).fetch1(
        "n_records"
    ) == 2
    assert len(covest.Traces()) == 2
    fetched = (covest.Traces & {"aod_scan_start_time": 1001}).fetch1()
    np.testing.assert_array_equal(fetched["trace_segments"], rows[0]["trace_segments"])


@pytest.mark.integration
def test_traces_round_trip(clean_schema):
    covest = clean_schema
    row = simulated_traces()
    covest.Traces.insert1(row)
    fetched = (covest.Traces & {"aod_scan_start_time": 1001}).fetch1()
    np.testing.assert_array_equal(fetched["trace_segments"], row["trace_segments"])
    assert fetched["high_repeats"]


@pytest.mark.integration
def test_active_cells_only_for_correlation_sets(clean_schema):
    covest = clean_schema
    covest.Traces.insert(
        [
            simulated_traces(scan=1),
            simulated_traces(scan=2, high_repeats=False, n_dirs=16),
        ]
    )
    covest.ActiveCells.populate()
    assert len(covest.ActiveCells()) == 1
    assert (covest.ActiveCells & {"aod_scan_start_time": 1001}).fetch1("ncells") == 110


@pytest.mark.integration
def test_cov_matrix_sample(clean_schema):
    covest = clean_schema
    covest.Traces.insert(
        [simulated_traces(scan=1), simulated_traces(scan=2, n_cells=90)]
    )
    covest.ActiveCells.populate()
    # sites with 100 cells or fewer are not estimated
    assert len(covest.CovMatrix.key_source & {"aod_scan_start_time": 1002}) == 0
    covest.CovMatrix.populate({"method": 0})
    assert len(covest.CovMatrix()) == 11
    R = (covest.CovMatrix & {"k": 1, "nfolds": 1, "method": 0}).fetch1("corr_matrix")
    np.testing.assert_allclose(np.diag(R), 1)
    losses = (covest.CovMatrix & "nfolds = 10").to_arrays("cv_loss")
    assert np.all(np.isfinite(losses))
