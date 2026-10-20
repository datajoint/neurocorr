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


def _source_record(session, scan, high_repeats):
    """A published record: a trace set with its original identifiers."""
    row = simulated_traces(
        site=scan % 10000, high_repeats=high_repeats, n_dirs=2 if high_repeats else 16
    )
    del row["session"], row["site"]
    return {
        **row,
        "subject_id": 16,
        "setup": 4,
        "session_start_time": session,
        "aod_scan_start_time": scan,
    }


# sessions 31234 and 42345 (numbered 1234 and 2345) have records in several files
SOURCE = {
    "traces-pre5-0000.mat": [
        _source_record(42345, 62001, True),
        _source_record(31234, 51002, False),
    ],
    "traces-pre5-0001.mat": [_source_record(31234, 51002, True)],
    "traces-pre5-0002.mat": [_source_record(31234, 51001, True)],
    "traces-pre5-0003.mat": [_source_record(42345, 62001, False)],
}


def _with_source(ingest, monkeypatch, tmp_path, source, fetched):
    """Serve each file from tmp_path, record which file each fetch asks for, and expect each
    file's own number of records."""
    paths = {name: write_mat(tmp_path / name, rows) for name, rows in source.items()}

    def fetch(file_name, sha256):
        fetched.append(file_name)
        return paths[file_name]

    monkeypatch.setattr(ingest, "fetch", fetch)
    monkeypatch.setattr(
        ingest,
        "SOURCE_FILES",
        [(name, sha, len(source[name])) for name, sha, _ in ingest.SOURCE_FILES],
    )


def _contents(covest):
    return (
        covest.Session.to_dicts(order_by="session"),
        covest.Site.to_dicts(order_by=("session", "site")),
        covest.TrialTraceSet.proj("session", "site", "high_repeats").to_dicts(
            order_by="trace_set"
        ),
    )


@pytest.mark.integration
def test_source_import_one_job_per_file(clean_schema, tmp_path, monkeypatch):
    ingest, covest = clean_schema.ingest, clean_schema.covest
    fetched = []
    _with_source(ingest, monkeypatch, tmp_path, SOURCE, fetched)

    files = ingest.SourceFile.to_dicts(order_by="source_file")
    assert [f["first_trace_set"] for f in files] == [1, 17, 42, 57]
    for file in files:
        fetched.clear()
        ingest.SourceImport.populate({"source_file": file["source_file"]})
        # each job reads its own file only
        assert fetched == [file["source_file"]]

    sessions, sites, sets = _contents(covest)
    # a session and a site take the last four digits of their start times
    assert [(s["session"], s["session_start_time"]) for s in sessions] == [
        (1234, 31234),
        (2345, 42345),
    ]
    assert [(s["session"], s["site"], s["aod_scan_start_time"]) for s in sites] == [
        (1234, 1001, 51001),
        (1234, 1002, 51002),
        (2345, 2001, 62001),
    ]
    # trace sets follow the files: each file from its first number, in file order
    assert [
        (s["trace_set"], s["session"], s["site"], bool(s["high_repeats"])) for s in sets
    ] == [
        (1, 2345, 2001, True),
        (2, 1234, 1002, False),
        (17, 1234, 1002, True),
        (42, 1234, 1001, True),
        (57, 2345, 2001, False),
    ]
    fetched_set = (covest.TrialTraceSet & {"trace_set": 17}).fetch1("trace_segments")
    np.testing.assert_array_equal(
        fetched_set, SOURCE["traces-pre5-0001.mat"][0]["trace_segments"]
    )


@pytest.mark.integration
def test_source_import_numbers_do_not_depend_on_job_order(
    clean_schema, tmp_path, monkeypatch
):
    import importlib
    import os

    import datajoint as dj

    results = []
    for order in (sorted(SOURCE), sorted(SOURCE, reverse=True)):
        prefix = os.environ.get("DATABASE_PREFIX", "")
        for name in ("ingest", "covest"):
            dj.Schema(prefix + name).drop(prompt=False)
        covest = importlib.reload(clean_schema.covest)
        ingest = importlib.reload(clean_schema.ingest)
        _with_source(ingest, monkeypatch, tmp_path, SOURCE, [])
        for name in order:
            ingest.SourceImport.populate({"source_file": name})
        results.append(_contents(covest))
    assert results[0] == results[1]


@pytest.mark.integration
def test_source_import_refuses_a_number_for_two_sessions(
    clean_schema, tmp_path, monkeypatch
):
    import datajoint as dj

    ingest = clean_schema.ingest
    # 31234 and 71234 share their last four digits
    source = {**SOURCE, "traces-pre5-0003.mat": [_source_record(71234, 81001, True)]}
    _with_source(ingest, monkeypatch, tmp_path, source, [])
    ingest.SourceImport.populate({"source_file": "traces-pre5-0002.mat"})
    with pytest.raises(dj.DataJointError, match="already session_start_time 31234"):
        ingest.SourceImport.populate({"source_file": "traces-pre5-0003.mat"})


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
