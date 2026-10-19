"""
Integration tests: the pipeline on a database, with simulated recordings.
"""

import numpy as np
import pytest

FIELDS = [
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


def simulated_traces(n_cells=110, n_dirs=2, n_trials=30, high_repeats=True, site=1):
    """A trial trace set's attributes, as the ingest inserts them, at a site of session 1."""
    rng = np.random.default_rng([site, high_repeats])
    n_bins = 4 if high_repeats else 3
    traces = rng.exponential(size=(n_bins, n_dirs, n_trials, n_cells)).astype(
        np.float32
    )
    traces[:, -1, -1] = np.nan
    return {
        "session": 1,
        "site": site,
        "high_repeats": high_repeats,
        "preprocess_method_num": 5,
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


def insert_trace_sets(covest, sets):
    """Insert trace sets with one session and a site for each."""
    covest.Session.insert1({"session": 1, "setup": 4, "session_start_time": 1000})
    for n, row in enumerate(sets, start=1):
        covest.Site.insert1(
            {
                "session": 1,
                "site": row["site"],
                "aod_scan_start_time": 1000 + row["site"],
            },
            skip_duplicates=True,
        )
        covest.TrialTraceSet.insert1({"trace_set": n, **row})


def write_mat(path, records):
    """A source file: a struct array ``tuples``, one element per record, as published."""
    from scipy.io import savemat

    tuples = np.zeros((len(records),), dtype=[(f, object) for f in FIELDS])
    for i, record in enumerate(records):
        tuples[i] = tuple(
            int(record[f]) if f == "high_repeats" else record[f] for f in FIELDS
        )
    savemat(path, {"tuples": tuples})
    return path


@pytest.mark.integration
def test_source_import_numbers_across_files(clean_schema, tmp_path, monkeypatch):
    ingest, covest = clean_schema.ingest, clean_schema.covest

    def record(session, scan, high_repeats):
        row = simulated_traces(
            site=scan - session,
            high_repeats=high_repeats,
            n_dirs=2 if high_repeats else 16,
        )
        del row["session"], row["site"]
        return {
            **row,
            "subject_id": 16,
            "setup": 4,
            "session_start_time": session,
            "aod_scan_start_time": scan,
        }

    # one session's sets spread over the files, out of order
    files = {
        "traces-pre5-0000.mat": [record(60000, 62001, True)],
        "traces-pre5-0001.mat": [record(50000, 51002, False)],
        "traces-pre5-0002.mat": [record(50000, 51002, True)],
        "traces-pre5-0003.mat": [record(50000, 51001, True)],
    }
    paths = {name: write_mat(tmp_path / name, rows) for name, rows in files.items()}
    monkeypatch.setattr(ingest, "fetch", lambda file_name, sha256: paths[file_name])

    assert len(ingest.SourceFile()) == 4
    ingest.SourceImport.populate()
    assert ingest.SourceImport.fetch1("n_records") == 4
    sessions = covest.Session.to_dicts(order_by="session")
    assert [(s["session"], s["session_start_time"]) for s in sessions] == [
        (1, 50000),
        (2, 60000),
    ]
    # a site takes its number in the original code: its scan start time mod 10000
    sites = covest.Site.to_dicts(order_by=("session", "site"))
    assert [(s["session"], s["site"], s["aod_scan_start_time"]) for s in sites] == [
        (1, 1001, 51001),
        (1, 1002, 51002),
        (2, 2001, 62001),
    ]
    sets = covest.TrialTraceSet.proj("session", "site", "high_repeats").to_dicts(
        order_by="trace_set"
    )
    assert [
        (s["trace_set"], s["session"], s["site"], bool(s["high_repeats"])) for s in sets
    ] == [
        (1, 1, 1001, True),
        (2, 1, 1002, False),
        (3, 1, 1002, True),
        (4, 2, 2001, True),
    ]
    fetched = (covest.TrialTraceSet & {"trace_set": 3}).fetch1("trace_segments")
    np.testing.assert_array_equal(
        fetched, files["traces-pre5-0002.mat"][0]["trace_segments"]
    )


@pytest.mark.integration
def test_trace_set_round_trip(clean_schema):
    covest = clean_schema.covest
    row = simulated_traces()
    insert_trace_sets(covest, [row])
    fetched = (covest.TrialTraceSet & {"trace_set": 1}).fetch1()
    np.testing.assert_array_equal(fetched["trace_segments"], row["trace_segments"])
    assert fetched["high_repeats"]
    # a site has one trace set of each kind
    with pytest.raises(Exception):
        covest.TrialTraceSet.insert1({**row, "trace_set": 2})


@pytest.mark.integration
def test_cell_selection_only_for_correlation_sets(clean_schema):
    covest = clean_schema.covest
    insert_trace_sets(
        covest,
        [
            simulated_traces(site=1),
            simulated_traces(site=2, high_repeats=False, n_dirs=16),
        ],
    )
    covest.CellSelection.populate()
    assert len(covest.CellSelection()) == 1
    assert (covest.CellSelection & {"trace_set": 1}).fetch1("ncells") == 110


@pytest.mark.integration
def test_cov_matrix_sample(clean_schema):
    covest = clean_schema.covest
    insert_trace_sets(
        covest, [simulated_traces(site=1), simulated_traces(site=2, n_cells=90)]
    )
    covest.CellSelection.populate()
    # sites with 100 cells or fewer are not estimated
    assert len(covest.CovMatrix.key_source & {"trace_set": 2}) == 0
    covest.CovMatrix.populate({"method": 0})
    assert len(covest.CovMatrix()) == 11
    R = (covest.CovMatrix & {"k": 1, "nfolds": 1, "method": 0}).fetch1("corr_matrix")
    np.testing.assert_allclose(np.diag(R), 1)
    losses = (covest.CovMatrix & "nfolds = 10").to_arrays("cv_loss")
    assert np.all(np.isfinite(losses))
