"""
Schema definitions: ``covest`` (the pipeline, from Session on) and ``ingest`` (the published
data files and their import into it).

Schema names take the prefix in the ``DATABASE_PREFIX`` environment variable, if set.
"""

from neurocorr.pipeline import covest, ingest

__all__ = ["covest", "ingest"]
