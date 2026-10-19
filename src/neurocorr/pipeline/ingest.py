"""
Ingest schema: the study's published data and the entities it holds.

    Source ──> SourceFile
       └─────> SourceImport ··· fills ···> Session ··> Site ··> TrialTraceSet

``SourceImport`` reads the published files of the original study and inserts the sessions,
the imaging sites, and their trial trace sets (fan-out ingestion). Each gets a simple number
for its key, in the order of its identifiers in the original database, which it keeps as
attributes. A dotted arrow above is a secondary reference: a trace set belongs to a site and
a site to a session, but neither is identified by it.

The published records give every session the same ``subject_id`` (16), although the study
imaged 14 mice: the identifier does not tell the animals apart, so it is not kept. The
paper's 27 analyzed sites come from 14 sessions, one per mouse.
"""

import os

import datajoint as dj

from neurocorr.ingest import COV_EST_COMMIT, SOURCE_FILES, fetch, read_records

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
    # The import of a source's sessions, sites, and trial trace sets
    -> Source
    ---
    n_records : int16
    """

    def make(self, key):
        """Download the source's files if needed, verify them, and insert their records.

        Each record of a file is one row of the original ``covest.Traces``: the trial traces of
        one imaging site for one stimulus set. Sessions, sites, and trace sets are numbered from
        1 in the order of their original identifiers (session start, scan start, then the
        tuning set before the correlation set), across all the files at once, so the numbers do
        not depend on the order the files are read in.
        """
        records = [
            record
            for file in (SourceFile & key).to_dicts()
            for record in read_records(fetch(file["source_file"], file["sha256"]))
        ]
        records.sort(
            key=lambda r: (r.session_start_time, r.aod_scan_start_time, r.high_repeats)
        )
        sessions = sorted({r.session_start_time for r in records})
        sites = sorted({(r.session_start_time, r.aod_scan_start_time) for r in records})
        session_of = {start: n for n, start in enumerate(sessions, start=1)}
        site_of = {scan: n for n, scan in enumerate(sites, start=1)}

        setup_of = {r.session_start_time: r.setup for r in records}
        Session.insert(
            {
                "session": session_of[start],
                "setup": setup_of[start],
                "session_start_time": start,
            }
            for start in sessions
        )
        Site.insert(
            {
                "site": site_of[(start, scan)],
                "session": session_of[start],
                "aod_scan_start_time": scan,
            }
            for start, scan in sites
        )
        TrialTraceSet.insert(
            {
                "trace_set": n,
                "site": site_of[(r.session_start_time, r.aod_scan_start_time)],
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
    site                : int16
    ---
    -> Session
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
    unique index (site, high_repeats)
    """
