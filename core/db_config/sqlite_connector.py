"""SQLite SQLDatabase connector helpers."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from core.db_config.source_registry import SQLSourceConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_sqlite_database_path(source: SQLSourceConfig) -> Path:
    """Resolve and validate the database file configured for a SQLite source."""
    env_name = source.env.get("path", "").strip()
    configured_path = os.getenv(env_name, "").strip() if env_name else ""
    path_value = configured_path or (source.default_database or "").strip()
    if not path_value:
        raise ValueError(f"Missing SQLite database path for source '{source.id}'")

    database_path = Path(path_value).expanduser()
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    database_path = database_path.resolve()

    if not database_path.exists():
        raise ValueError(
            f"SQLite database file for source '{source.id}' does not exist: {database_path}"
        )
    if not database_path.is_file():
        raise ValueError(
            f"SQLite database path for source '{source.id}' is not a file: {database_path}"
        )
    return database_path


def create_sqlite_sql_database(
    source: SQLSourceConfig, include_tables: list[str]
) -> SQLDatabase:
    """Create SQLDatabase for a SQLite source and table subset."""
    database_path = resolve_sqlite_database_path(source)
    url = URL.create(drivername="sqlite", database=str(database_path))
    engine = create_engine(url)

    return SQLDatabase(
        engine,
        include_tables=include_tables,
        sample_rows_in_table_info=0,
    )
