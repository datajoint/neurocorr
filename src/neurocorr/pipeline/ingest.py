"""
Ingest schema: the published data files of the original study, and their import.

    Source ──> SourceFile
       └─────> SourceImport ··· fills ···> covest.Session, covest.Site, covest.TrialTraceSet

The schema handles the files only. ``SourceImport`` reads them and inserts the sessions, the
imaging sites, and their trial trace sets into the pipeline's ``covest`` schema, which has no
reference back to it (fan-out ingestion): the pipeline starts at ``covest.Session``.

Sessions are numbered from 1 in the order of their start times. A site takes the number the
original code and figure scripts use for it, the last four digits of its scan start time
(``mod(aod_scan_start_time, 10000)``; the example site of Fig 3 is 2031). Trace sets are numbered
from 1 in the order of their session and scan, the tuning set before the correlation set. The
original identifiers stay on the rows.

The published records give every session the same ``subject_id`` (16), although the study
imaged 14 mice: the identifier does not tell the animals apart, so it is not kept. The paper's
27 analyzed sites come from 14 sessions, one per mouse.
"""

import os

import datajoint as dj

from neurocorr.ingest import COV_EST_COMMIT, SOURCE_FILES, fetch, read_records
from neurocorr.pipeline.covest import Session, Site, TrialTraceSet

schema = dj.Schema(os.environ.get("DATABASE_PREFIX", "") + "ingest")


@schema
class Source(dj.Lookup):
    definition = """
    # Published data of the original study: the data files of a commit of github.com/atlab/cov-est
    source : varchar(16)
    ---
    commit : char(40)
    """
    contents = [{"source": "cov-est", "commit": COV_EST_COMMIT}]


@schema
class SourceFile(dj.Lookup):
    definition = """
    # One data file of a source, verified by its SHA-256
    -> Source
    source_file : varchar(24)
    ---
    sha256      : char(64)
    """
    contents = [
        {"source": "cov-est", "source_file": name, "sha256": sha256}
        for name, sha256 in SOURCE_FILES
    ]


@schema
class SourceImport(dj.Ingest):
    definition = """
    # The import of a source's sessions, sites, and trial trace sets into the pipeline
    -> Source
    ---
    n_records : int16
    """

    def make(self, key):
        """Download the source's files if needed, verify them, and insert their records.

        Each record of a file is one row of the original ``covest.Traces``: the trial traces of
        one imaging site for one stimulus set. All the files are read before anything is
        numbered, so the numbers do not depend on the order the files are read in.
        """
        records = [
            record
            for file in (SourceFile & key).to_dicts()
            for record in read_records(fetch(file["source_file"], file["sha256"]))
        ]
        records.sort(
            key=lambda r: (r.session_start_time, r.aod_scan_start_time, r.high_repeats)
        )
        session_of = {
            start: n
            for n, start in enumerate(
                sorted({r.session_start_time for r in records}), start=1
            )
        }
        # scan -> (session, site), the site numbered as the original code numbers it
        site_of = {
            r.aod_scan_start_time: (
                session_of[r.session_start_time],
                r.aod_scan_start_time % 10000,
            )
            for r in records
        }

        setup_of = {r.session_start_time: r.setup for r in records}
        Session.insert(
            {"session": n, "setup": setup_of[start], "session_start_time": start}
            for start, n in session_of.items()
        )
        Site.insert(
            {"session": session, "site": site, "aod_scan_start_time": scan}
            for scan, (session, site) in site_of.items()
        )
        TrialTraceSet.insert(
            {
                "trace_set": n,
                "session": site_of[r.aod_scan_start_time][0],
                "site": site_of[r.aod_scan_start_time][1],
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
            for n, r in enumerate(records, start=1)
        )
        self.insert1({**key, "n_records": len(records)})
