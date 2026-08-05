"""ClickHouse SQLDatabase connector helpers."""

from __future__ import annotations

import os

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from core.db_config.source_registry import SQLSourceConfig


def create_clickhouse_sql_database(
    source: SQLSourceConfig, include_tables: list[str]
) -> SQLDatabase:
    """Create SQLDatabase for a ClickHouse source and table subset."""
    host = read_required_env(source, "host")
    port_value = read_required_env(source, "port")
    user = read_required_env(source, "user")
    password = read_required_env(source, "password")

    url = URL.create(
        drivername="clickhouse+http",
        username=user,
        password=password,
        host=host,
        port=int(port_value),
        database=source.default_database or "default",
    )
    engine = create_engine(url)

    return SQLDatabase(
        engine,
        include_tables=include_tables,
        sample_rows_in_table_info=0,
    )


def read_required_env(source: SQLSourceConfig, env_key: str) -> str:
    """Read one required env var value declared in a source config."""
    env_name = source.env.get(env_key, "").strip()
    if not env_name:
        raise ValueError(f"Missing env mapping '{env_key}' for source '{source.id}'")
    env_value = os.getenv(env_name, "").strip()
    if not env_value:
        raise ValueError(
            f"Environment variable '{env_name}' is required for source '{source.id}'"
        )
    return env_value
