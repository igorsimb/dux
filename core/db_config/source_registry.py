"""Load and validate configured SQL sources for database routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ai.sql_dialect import SQLDialect

SQL_SOURCES_PATH = Path(__file__).resolve().parent / "sql_sources.json"


@dataclass(frozen=True)
class SQLSourceConfig:
    """Typed configuration for one SQL source entry."""

    id: str
    dialect: SQLDialect
    driver: str
    env: dict[str, str]
    default_database: str | None = None


def load_sql_sources(path: str | Path | None = None) -> list[SQLSourceConfig]:
    """Return all configured SQL sources after schema validation.

    Example output:
    [
        SQLSourceConfig(
            id="clickhouse_default",
            dialect=SQLDialect.CLICKHOUSE,
            driver="clickhouse",
            env={"host": "CLICKHOUSE_HOST"},
            default_database="default",
        ),
        SQLSourceConfig(
            id="mssql_default",
            dialect=SQLDialect.MSSQL,
            driver="mssql",
            env={"host": "MSSQL_HOST", "database": "MSSQL_DATABASE"},
            default_database=None,
        ),
    ]
    """
    config_path = Path(path) if path is not None else SQL_SOURCES_PATH
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON root in {config_path}: expected object")
    if "sources" not in raw:
        raise ValueError(f"Missing required key 'sources' in {config_path}")

    sources_raw = raw["sources"]
    if not isinstance(sources_raw, list):
        raise ValueError(f"Invalid 'sources' type in {config_path}: expected list")

    sources: list[SQLSourceConfig] = []
    seen_ids: set[str] = set()

    for index, source_raw in enumerate(sources_raw):
        if not isinstance(source_raw, dict):
            raise ValueError(
                f"Invalid source entry at index {index} in {config_path}: expected object"
            )

        source_id = str(source_raw.get("id", "")).strip()
        if not source_id:
            raise ValueError(
                f"Missing required source field 'id' at index {index} in {config_path}"
            )
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source id '{source_id}' in {config_path}")
        seen_ids.add(source_id)

        dialect_raw = source_raw.get("dialect")
        if not isinstance(dialect_raw, str):
            raise ValueError(
                f"Missing required source field 'dialect' for source '{source_id}'"
            )
        dialect_name: str = dialect_raw
        dialect = SQLDialect.from_raw(dialect_name)

        driver = str(source_raw.get("driver", "")).strip()
        if not driver:
            raise ValueError(
                f"Missing required source field 'driver' for source '{source_id}'"
            )

        env_raw = source_raw.get("env")
        if not isinstance(env_raw, dict):
            raise ValueError(
                f"Invalid env config for source '{source_id}': expected object"
            )

        env: dict[str, str] = {}
        for key, value in env_raw.items():
            env_key = str(key).strip()
            env_value = str(value).strip()
            if not env_key:
                raise ValueError(
                    f"Invalid env key for source '{source_id}': key cannot be empty"
                )
            if not env_value:
                raise ValueError(
                    f"Invalid env value for source '{source_id}': value for key '{env_key}' cannot be empty"
                )
            env[env_key] = env_value

        default_database_raw = source_raw.get("default_database")
        default_database = (
            str(default_database_raw).strip() if default_database_raw else None
        )

        sources.append(
            SQLSourceConfig(
                id=source_id,
                dialect=dialect,
                driver=driver,
                env=env,
                default_database=default_database,
            )
        )

    return sources


def get_source_by_id(source_id: str, path: str | Path | None = None) -> SQLSourceConfig:
    """Return one source config by id or raise if it is not configured.

    Example output:
    SQLSourceConfig(
        id="mssql_default",
        dialect=SQLDialect.MSSQL,
        driver="mssql",
        env={"host": "MSSQL_HOST", "database": "MSSQL_DATABASE"},
        default_database=None,
    )
    """
    source_id_normalized = source_id.strip()
    for source in load_sql_sources(path):
        if source.id == source_id_normalized:
            return source
    raise ValueError(f"Source not found: {source_id_normalized}")


def get_sources_by_dialect(
    dialect: SQLDialect | str, path: str | Path | None = None
) -> list[SQLSourceConfig]:
    """Return all configured sources for a dialect.

    Example output for `dialect=SQLDialect.MSSQL`:
    [
        SQLSourceConfig(
            id="mssql_default",
            dialect=SQLDialect.MSSQL,
            driver="mssql",
            env={"host": "MSSQL_HOST", "database": "MSSQL_DATABASE"},
            default_database=None,
        )
    ]
    """
    dialect_value = SQLDialect.from_raw(dialect)
    return [
        source for source in load_sql_sources(path) if source.dialect == dialect_value
    ]
