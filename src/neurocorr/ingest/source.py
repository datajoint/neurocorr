"""
The original study's data files, and reading them.

The recordings of Yatsenko et al. (2015) are published with the study's MATLAB code in
https://github.com/atlab/cov-est (MIT License, (c) 2014 Tolias Lab) as four MATLAB v5
files. Each holds a struct array ``tuples``: one element per (scan, stimulus protocol),
with the traces and the metadata of that recording. Files are fetched from a pinned
commit, verified by SHA-256, and cached locally.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The commit of atlab/cov-est that last changed data/
COV_EST_COMMIT = "eb60af1033931282d291a87120f85aa7731a70fb"
COV_EST_RAW = f"https://raw.githubusercontent.com/atlab/cov-est/{COV_EST_COMMIT}/data"

# file name, SHA-256, and number of records (30.6, 100.2, 66.6 and 30.8 MB); a file's records
# are fixed by its hash
SOURCE_FILES = [
    (
        "traces-pre5-0000.mat",
        "95f4cb2b09064eb5e2f702394304eb1c23540c295b8677ffc7a3dbbc018607ed",
        16,
    ),
    (
        "traces-pre5-0001.mat",
        "c05004aff7c858fc717cfd22ca36003b4ff11f196d9d518ca4bf1f165cd92592",
        25,
    ),
    (
        "traces-pre5-0002.mat",
        "5c8281698637195633046b6c944e3fe6c14bb35627126b5c523ba7ed97b211bf",
        15,
    ),
    (
        "traces-pre5-0003.mat",
        "c337f9a4819f522a6be0ddb1d69a0325414a4cb8bb27651ea9215763fc973c42",
        4,
    ),
]


def data_dir() -> Path:
    """Directory holding the source files: ``NEUROCORR_DATA_DIR``, else ``~/.cache/neurocorr``."""
    return Path(
        os.environ.get("NEUROCORR_DATA_DIR", Path.home() / ".cache" / "neurocorr")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(file_name: str, expected_sha256: str) -> Path:
    """Return the local path of a source file, downloading and verifying it if needed."""
    path = data_dir() / file_name
    url = f"{COV_EST_RAW}/{file_name}"
    if path.exists() and sha256(path) == expected_sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise ValueError(
            f"{file_name}: SHA-256 {actual} does not match {expected_sha256}"
        )
    partial.replace(path)
    return path


@dataclass
class Record:
    """One element of a source file's ``tuples``: the recording of one protocol in one scan."""

    setup: int
    session_start_time: int
    aod_scan_start_time: int
    preprocess_method_num: int
    high_repeats: bool
    latency_ms: int
    bin_ms: int
    evoked_bins: int
    cell_nums: np.ndarray  # (cells,)
    cell_xyz: np.ndarray  # (cells, 3), µm
    directions: np.ndarray  # (conditions,), degrees
    n_trials: np.ndarray  # (conditions,)
    traces: np.ndarray  # (bins, conditions, trials, cells), float32, NaN-padded


def read_records(path: Path) -> list[Record]:
    """Read every record of a source file."""
    from scipy.io import loadmat

    tuples = np.atleast_1d(
        loadmat(path, squeeze_me=True, struct_as_record=False)["tuples"]
    )
    records = []
    for t in tuples:
        traces = np.asarray(t.trace_segments, dtype=np.float32)
        if traces.ndim == 3:  # a single condition was squeezed out
            traces = traces[:, None]
        records.append(
            Record(
                setup=int(t.setup),
                session_start_time=int(t.session_start_time),
                aod_scan_start_time=int(t.aod_scan_start_time),
                preprocess_method_num=int(t.preprocess_method_num),
                high_repeats=bool(t.high_repeats),
                latency_ms=int(t.latency_ms),
                bin_ms=int(t.bin_ms),
                evoked_bins=int(t.evoked_bins),
                cell_nums=np.atleast_1d(t.cellnums).astype(np.int32),
                cell_xyz=np.atleast_2d(t.cell_xyz).astype(np.float64),
                directions=np.atleast_1d(t.directions).astype(np.float64),
                n_trials=np.atleast_1d(t.ntrials).astype(np.int32),
                traces=traces,
            )
        )
    return records
