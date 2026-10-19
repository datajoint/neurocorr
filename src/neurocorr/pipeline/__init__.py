"""
Schema definitions: ``ingest`` (the published data) and ``covest`` (the analysis).

Schema names take the prefix in the ``DATABASE_PREFIX`` environment variable, if set.
"""

from neurocorr.pipeline import covest, ingest

__all__ = ["covest", "ingest"]
