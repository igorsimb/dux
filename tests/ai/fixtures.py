import pytest


@pytest.fixture
def supported_dialects() -> tuple[str, ...]:
    return ("sqlite", "postgres", "clickhouse", "tsql")
