"""
Integration test fixtures.

Uses testcontainers to start a PostgreSQL database for the tests.
"""

import os

import pytest


@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL container for the test session."""
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
    """Declare the schema afresh for each test, and drop it afterwards."""
    import importlib

    import datajoint as dj

    from neurocorr.pipeline import covest

    name = os.environ.get("DATABASE_PREFIX", "") + "covest"
    dj.Schema(name).drop(prompt=False)
    # `@schema` declares tables at import time; reload to declare them again
    importlib.reload(covest)
    yield covest
    dj.Schema(name).drop(prompt=False)
