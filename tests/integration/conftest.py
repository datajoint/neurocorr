"""
Integration test fixtures.

Uses testcontainers to start a PostgreSQL database for the tests.
"""

import os

import pytest


@pytest.fixture(scope="session")
def postgres_container(tmp_path_factory):
    """Start a PostgreSQL container for the test session, with a file store for arrays."""
    import datajoint as dj

    dj.config.stores = {
        "default": "main",
        "main": {"protocol": "file", "location": str(tmp_path_factory.mktemp("store"))},
    }
    if os.environ.get("DJ_USE_EXTERNAL_DB"):
        yield None
        return

    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:
        from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16") as postgres:
        os.environ["DJ_HOST"] = postgres.get_container_host_ip()
        os.environ["DJ_USER"] = "test"
        os.environ["DJ_PASS"] = "test"
        os.environ["DJ_PORT"] = str(postgres.get_exposed_port(5432))
        yield postgres


@pytest.fixture
def clean_schema(postgres_container):
    """Declare both schemas afresh for each test, and drop them afterwards."""
    import importlib
    from types import SimpleNamespace

    import datajoint as dj

    from neurocorr.pipeline import covest, ingest

    prefix = os.environ.get("DATABASE_PREFIX", "")

    def drop():
        for name in ("ingest", "covest"):
            dj.Schema(prefix + name).drop(prompt=False)

    drop()
    # `@schema` declares tables at import time; reload to declare them again (the ingest
    # module imports the pipeline's tables, so the pipeline first)
    importlib.reload(covest)
    importlib.reload(ingest)
    yield SimpleNamespace(ingest=ingest, covest=covest)
    drop()
