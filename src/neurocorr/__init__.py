"""
neurocorr: improved estimation and interpretation of correlations in neural circuits.

A DataJoint pipeline that reproduces the analysis of Yatsenko, Josić, Ecker,
Froudarakis, Cotton & Tolias (2015), PLOS Comput Biol 11(3):e1004083,
doi:10.1371/journal.pcbi.1004083, translated from its MATLAB code and data in
https://github.com/atlab/cov-est (MIT License, (c) 2014 Tolias Lab).

Structure
---------
- neurocorr.pipeline: the schema (``covest``)
- neurocorr.cove: covariance estimators and cross-validation
- neurocorr.ingest: the original data files: fetching, verifying, and reading them

Example
-------
>>> from neurocorr.pipeline import covest
>>> covest.TrialTraceSet()
"""

__version__ = "0.1.0"

__all__ = ["__version__", "cove", "ingest", "pipeline"]
