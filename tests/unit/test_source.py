"""
Unit tests for reading the original data files.
"""

import numpy as np
import pytest
from scipy.io import savemat

from neurocorr.ingest import source


@pytest.fixture
def mat_file(tmp_path):
    """A file with the layout of the cov-est data: a struct array `tuples`."""
    rng = np.random.default_rng(0)
    dtype = [
        (name, object)
        for name in [
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
    ]
    tuples = np.zeros((2,), dtype=dtype)
    for i, (high_repeats, n_dirs, n_bins) in enumerate([(0, 16, 3), (1, 2, 14)]):
        traces = rng.exponential(size=(n_bins, n_dirs, 7, 4)).astype(np.float32)
        traces[:, -1, -1] = np.nan  # the last direction has one trial fewer
        tuples[i] = (
            16,
            4,
            3453728398828,
            3453748112031,
            5,
            high_repeats,
            20,
            150.0,
            8 if high_repeats else 4,
            4,
            rng.uniform(-99, 99, size=(4, 3)),
            np.array([1, 2, 5, 9], dtype=np.uint8),
            n_dirs,
            np.linspace(0, 360, n_dirs, endpoint=False),
            np.array([7] * (n_dirs - 1) + [6], dtype=np.uint8),
            traces,
        )
    path = tmp_path / "traces.mat"
    savemat(path, {"tuples": tuples})
    return path


def test_read_records(mat_file):
    tuning, correlation = source.read_records(mat_file)
    assert not tuning.high_repeats and correlation.high_repeats
    assert tuning.traces.shape == (3, 16, 7, 4) and correlation.traces.shape == (
        14,
        2,
        7,
        4,
    )
    assert correlation.traces.dtype == np.float32
    np.testing.assert_array_equal(correlation.cell_nums, [1, 2, 5, 9])
    assert correlation.n_trials.dtype == np.int32 and correlation.n_trials.sum() == 13
    assert correlation.aod_scan_start_time == 3453748112031


def test_fetch_uses_a_verified_local_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROCORR_DATA_DIR", str(tmp_path))
    (tmp_path / "x.mat").write_bytes(b"data")
    digest = source.sha256(tmp_path / "x.mat")
    assert source.fetch("x.mat", digest) == tmp_path / "x.mat"


def test_source_files_are_pinned():
    assert len(source.SOURCE_FILES) == 4
    assert source.COV_EST_COMMIT in source.COV_EST_RAW
    assert all(len(sha) == 64 for _, sha, _ in source.SOURCE_FILES)
    assert sum(n for _, _, n in source.SOURCE_FILES) == 60
