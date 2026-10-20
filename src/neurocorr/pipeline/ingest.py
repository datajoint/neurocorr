"""
Ingest schema: the published data files of the original study, and their import.

    SourceFile ──> SourceImport ··· fills ···> covest.Session, covest.Site, covest.TrialTraceSet

The schema handles the files only: the four data files published with the original code, at
a pinned commit of github.com/atlab/cov-est (``neurocorr.ingest.COV_EST_COMMIT``). Each file is
one ``SourceImport`` job, which reads that file alone and inserts its sessions, imaging sites,
and trial trace sets into the pipeline's ``covest`` schema. The pipeline has no reference back
to it (fan-out ingestion): it starts at ``covest.Session``.

Every number is fixed by the record itself or by its file, so the jobs can run in any order:

- a session is the last four digits of its start time, as the original code labels sites
  (distinct for the study's 15 sessions); a session with records in two files is inserted by
  whichever job comes first, with the same number;
- a site is the last four digits of its scan start time, within its session;
- a trace set is numbered in the order of the published files: a file's first trace set is
  recorded in ``SourceFile``, and its records follow in file order.

The original identifiers stay on the rows. The published records give every session the same
``subject_id`` (16), although the study imaged 14 mice: the identifier does not tell the
animals apart, so it is not kept. The paper's 27 analyzed sites come from 14 sessions, one per
mouse.
"""

import os
from itertools import accumulate

import datajoint as dj

from neurocorr.ingest import SOURCE_FILES, fetch, read_records
from neurocorr.pipeline.covest import Session, Site, TrialTraceSet

schema = dj.Schema(os.environ.get("DATABASE_PREFIX", "") + "ingest")

# a number fixed by the last four digits of an identifier in the original database
LABEL_MOD = 10000


@schema
class SourceFile(dj.Lookup):
    definition = """
    # A data file published with the original code, verified by its SHA-256
    source_file     : varchar(24)
    ---
    sha256          : char(64)
    first_trace_set : int16    # the number of the file's first trace set; its records follow in order
    """
    contents = [
        {"source_file": name, "sha256": sha256, "first_trace_set": first}
        for (name, sha256, _), first in zip(
            SOURCE_FILES,
            accumulate((n for *_, n in SOURCE_FILES[:-1]), initial=1),
        )
    ]


@schema
class SourceImport(dj.Ingest):
    definition = """
    # The import of one data file's sessions, sites, and trial trace sets into the pipeline
    -> SourceFile
    ---
    n_records : int16
    """

    def make(self, key):
        """Download the file if needed, verify it, and insert its records.

        Each record is one row of the original ``covest.Traces``: the trial traces of one
        imaging site for one stimulus set. Only this file is read. A session or site another
        file already inserted is kept, after checking that it is the same one.
        """
        file = (SourceFile & key).fetch1()
        records = read_records(fetch(file["source_file"], file["sha256"]))
        expected = {name: n for name, _, n in SOURCE_FILES}.get(file["source_file"])
        if expected is not None and len(records) != expected:
            raise dj.DataJointError(
                f"{file['source_file']}: {len(records)} records, expected {expected}"
            )

        sessions = _unique(
            [
                {
                    "session": r.session_start_time % LABEL_MOD,
                    "setup": r.setup,
                    "session_start_time": r.session_start_time,
                }
                for r in records
            ],
            ["session"],
            "session_start_time",
        )
        sites = _unique(
            [
                {
                    "session": r.session_start_time % LABEL_MOD,
                    "site": r.aod_scan_start_time % LABEL_MOD,
                    "aod_scan_start_time": r.aod_scan_start_time,
                }
                for r in records
            ],
            ["session", "site"],
            "aod_scan_start_time",
        )
        _insert_or_check(Session, sessions, ["session"], "session_start_time")
        _insert_or_check(Site, sites, ["session", "site"], "aod_scan_start_time")
        TrialTraceSet.insert(
            {
                "trace_set": file["first_trace_set"] + i,
                "session": r.session_start_time % LABEL_MOD,
                "site": r.aod_scan_start_time % LABEL_MOD,
                "high_repeats": r.high_repeats,
                "preprocess_method_num": r.preprocess_method_num,
                "latency_ms": r.latency_ms,
                "bin_ms": r.bin_ms,
                "evoked_bins": r.evoked_bins,
                "nneurons": len(r.cell_nums),
                "cell_xyz": r.cell_xyz,
                "cellnums": r.cell_nums,
                "ndirs": len(r.directions),
                "directions": r.directions,
                "ntrials": r.n_trials,
                "trace_segments": r.traces,
            }
            for i, r in enumerate(records)
        )
        self.insert1({**key, "n_records": len(records)})


def _unique(rows: list[dict], key: list[str], identity: str) -> list[dict]:
    """The distinct rows, refusing two identities that share a number."""
    by_key: dict[tuple, dict] = {}
    for row in rows:
        k = tuple(row[name] for name in key)
        if k in by_key and by_key[k][identity] != row[identity]:
            raise dj.DataJointError(
                f"{dict(zip(key, k))} would number both {identity} "
                f"{by_key[k][identity]} and {row[identity]}"
            )
        by_key[k] = row
    return list(by_key.values())


def _insert_or_check(table, rows: list[dict], key: list[str], identity: str) -> None:
    """Insert the rows not yet present (another job may insert the same row meanwhile), then
    check that each present row has the same identity: a number never names two of them."""
    for row in rows:
        table.insert1(row, skip_duplicates=True)
        present = (table & {name: row[name] for name in key}).fetch1(identity)
        if present != row[identity]:
            raise dj.DataJointError(
                f"{table.__name__} {dict((n, row[n]) for n in key)} is already "
                f"{identity} {present}, not {row[identity]}"
            )
