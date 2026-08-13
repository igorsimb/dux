"""SQL tool utility helpers for dialect-aware config loading and guarded execution."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import sqlglot

from ai.sql_dialect import SQLDialect
from ai.sql_guard import GuardedQuerySQLDatabaseTool
from core.db_config.source_registry import (
    SQLSourceConfig,
    get_source_by_id,
    load_sql_sources,
)

TABLE_DESCRIPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "table_descriptions.json"
)
TABLE_METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "table_metadata.json"
)
CLICKHOUSE_PARAM_RE = re.compile(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[^{}]+\}")
RUN_QUERY_TOOLS: dict[str, Any] = {}
RUN_QUERY_TOOL_FACTORIES: dict[str, Callable[[], Any]] = {}


def load_dialect_and_tables(
    path: str | Path, expected_tables_type: type
) -> tuple[SQLDialect, Any]:
    """Load wrapped table config, validate schema, and return typed dialect and table payload.

    This validates both required top-level keys (`sql_dialect`, `tables`) and enforces the expected
    type for `tables` so malformed config fails fast with actionable errors.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON root in {path}: expected object")
    if "sql_dialect" not in raw:
        raise ValueError(f"Missing required key 'sql_dialect' in {path}")
    if "tables" not in raw:
        raise ValueError(f"Missing required key 'tables' in {path}")

    dialect = SQLDialect.from_raw(raw["sql_dialect"])
    tables = raw["tables"]
    if not isinstance(tables, expected_tables_type):
        expected_name = expected_tables_type.__name__
        actual_name = type(tables).__name__
        raise ValueError(
            f"Invalid 'tables' type in {path}: expected {expected_name}, got {actual_name}"
        )
    return dialect, tables


@lru_cache(maxsize=1)
def get_configured_sql_dialect() -> SQLDialect:
    """Return one active SQL dialect when allowlisted tables use a single dialect.

    This helper is transitional for components that still operate with one dialect context.
    It derives dialect from allowlisted table rows in `table_descriptions.json`.
    """
    rows = load_table_description_rows()
    dialects = {
        SQLDialect.from_raw(str(row["sql_dialect"]))
        for row in rows
        if bool(row.get("allowed"))
    }
    if not dialects:
        raise ValueError("No allowlisted tables configured with sql_dialect")
    if len(dialects) > 1:
        raise ValueError(
            "Multiple SQL dialects configured in allowlisted tables. "
            "Single-dialect helpers cannot choose one."
        )
    return next(iter(dialects))


@lru_cache(maxsize=1)
def get_configured_source_id() -> str:
    """Return one active source id when allowlisted tables use a single source."""
    rows = load_table_description_rows()
    source_ids = {str(row["source"]) for row in rows if bool(row.get("allowed"))}
    if not source_ids:
        raise ValueError("No allowlisted tables configured with source")
    if len(source_ids) > 1:
        raise ValueError(
            "Multiple sources configured in allowlisted tables. "
            "Single-source helpers cannot choose one."
        )
    return next(iter(source_ids))


@lru_cache(maxsize=1)
def load_allowed_tables() -> set[str]:
    rows = load_table_description_rows()
    allowed_tables: set[str] = set()
    for row in rows:
        if not bool(row.get("allowed")):
            continue
        table_name = str(row.get("table", "")).strip().lower()
        if table_name:
            allowed_tables.add(table_name)
    return allowed_tables


@lru_cache(maxsize=1)
def load_allowed_source_candidates() -> list[tuple[str, SQLDialect]]:
    """Return unique allowlisted `(source_id, dialect)` routing candidates.

    Candidates are derived from allowlisted rows in `ai/table_descriptions.json` and are used
    by `validate_sql` to test one query against each possible backend route.

    Example output:
    [
        ("clickhouse_default", SQLDialect.CLICKHOUSE),
        ("mssql_default", SQLDialect.MSSQL),
    ]
    """
    rows = load_table_description_rows()
    candidates: list[tuple[str, SQLDialect]] = []
    seen: set[tuple[str, SQLDialect]] = set()

    for row in rows:
        if not bool(row.get("allowed")):
            continue
        source_id = str(row.get("source", "")).strip()
        dialect = SQLDialect.from_raw(str(row.get("sql_dialect", "")).strip())
        candidate = (source_id, dialect)
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)

    return candidates


@lru_cache(maxsize=64)
def load_allowed_tables_for_source(source_id: str) -> set[str]:
    """Return allowlisted full table names for one source."""
    rows = load_table_description_rows()
    source_id_normalized = source_id.strip()
    allowed_tables: set[str] = set()

    for row in rows:
        if not bool(row.get("allowed")):
            continue
        if str(row.get("source", "")).strip() != source_id_normalized:
            continue
        table_name = str(row.get("table", "")).strip().lower()
        if table_name:
            allowed_tables.add(table_name)

    return allowed_tables


def build_short_name_index(allowed_full_tables: set[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for full_name in sorted(allowed_full_tables):
        _, separator, short_name = full_name.partition(".")
        if not separator:
            short_name = full_name
        index.setdefault(short_name, []).append(full_name)
    return index


@lru_cache(maxsize=1)
def get_short_name_index() -> dict[str, list[str]]:
    return build_short_name_index(load_allowed_tables())


@lru_cache(maxsize=64)
def get_short_name_index_for_source(source_id: str) -> dict[str, list[str]]:
    """Return short table-name index for one source allowlist."""
    return build_short_name_index(load_allowed_tables_for_source(source_id))


def build_guarded_query_tool(db: Any) -> GuardedQuerySQLDatabaseTool:
    """Build the backend-only guarded SQL query tool for one database."""
    return GuardedQuerySQLDatabaseTool(db=db)


def set_run_query_tool_for_source(source_id: str, tool: Any) -> None:
    """Register one sql_db_query tool for a source id."""
    RUN_QUERY_TOOLS[source_id] = tool
    RUN_QUERY_TOOL_FACTORIES.pop(source_id, None)


def set_run_query_tool_factory_for_source(
    source_id: str, factory: Callable[[], Any]
) -> None:
    """Register a lazy sql_db_query tool factory for a source id."""
    RUN_QUERY_TOOL_FACTORIES[source_id] = factory
    RUN_QUERY_TOOLS.pop(source_id, None)


def clear_run_query_tools() -> None:
    """Clear all registered sql_db_query tools."""
    RUN_QUERY_TOOLS.clear()
    RUN_QUERY_TOOL_FACTORIES.clear()


def get_run_query_tool_for_source(source_id: str) -> Any | None:
    """Return sql_db_query tool for source id, creating it lazily when needed."""
    tool = RUN_QUERY_TOOLS.get(source_id)
    if tool is not None:
        return tool

    factory = RUN_QUERY_TOOL_FACTORIES.get(source_id)
    if factory is None:
        return None

    tool = factory()
    RUN_QUERY_TOOLS[source_id] = tool
    return tool


@lru_cache(maxsize=1)
def load_sql_source_configs() -> list[SQLSourceConfig]:
    """Return configured SQL sources for connection routing."""
    return load_sql_sources()


def get_sql_source_config(source_id: str) -> SQLSourceConfig:
    """Return one SQL source config by source id."""
    return get_source_by_id(source_id)


def extract_clickhouse_placeholders(query: str) -> list[str]:
    seen: set[str] = set()
    placeholders: list[str] = []
    for match in CLICKHOUSE_PARAM_RE.finditer(query or ""):
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        placeholders.append(name)
    return placeholders


def format_sql_for_log(query: str, dialect: SQLDialect | str | None = None) -> str:
    try:
        dialect_value = (
            SQLDialect.from_raw(dialect) if dialect else get_configured_sql_dialect()
        )
        return sqlglot.parse_one(query, read=dialect_value).sql(
            dialect=dialect_value, pretty=True
        )
    except Exception:
        return query


@lru_cache(maxsize=1)
def load_table_descriptions() -> list[dict[str, object]]:
    rows = load_table_description_rows()
    return [row for row in rows if row.get("allowed")]


@lru_cache(maxsize=1)
def load_table_metadata() -> dict[str, dict[str, object]]:
    metadata_by_table = load_metadata_table_map()
    allowed_tables = load_allowed_tables()
    return {
        name: meta
        for name, meta in metadata_by_table.items()
        if isinstance(name, str) and name.strip().lower() in allowed_tables
    }


def load_table_description_rows(
    path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Load table description rows and validate required routing fields per row."""
    config_path = Path(path) if path is not None else Path(TABLE_DESCRIPTIONS_PATH)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON root in {config_path}: expected object")

    rows = raw.get("tables")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid 'tables' type in {config_path}: expected list")

    source_by_id = {source.id: source for source in load_sql_source_configs()}
    validated: list[dict[str, object]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid table row at index {index} in {config_path}: expected object"
            )

        table_name = str(row.get("table", "")).strip()
        if not table_name:
            raise ValueError(
                f"Missing required field 'table' at index {index} in {config_path}"
            )

        source_id = str(row.get("source", "")).strip()
        if not source_id:
            raise ValueError(
                f"Missing required field 'source' for table '{table_name}' in {config_path}"
            )
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(
                f"Unknown source '{source_id}' for table '{table_name}' in {config_path}"
            )

        sql_dialect_raw = str(row.get("sql_dialect", "")).strip()
        if not sql_dialect_raw:
            raise ValueError(
                f"Missing required field 'sql_dialect' for table '{table_name}' in {config_path}"
            )
        sql_dialect = SQLDialect.from_raw(sql_dialect_raw)
        if source.dialect != sql_dialect:
            raise ValueError(
                f"Dialect mismatch for table '{table_name}': source '{source_id}' uses {source.dialect}, "
                f"table row declares {sql_dialect}"
            )

        normalized = dict(row)
        normalized["table"] = table_name
        normalized["source"] = source_id
        normalized["sql_dialect"] = str(sql_dialect)
        validated.append(normalized)

    return validated


def load_metadata_table_map(
    path: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Load metadata table map from table_metadata.json"""
    metadata_path = Path(path) if path is not None else Path(TABLE_METADATA_PATH)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON root in {metadata_path}: expected object")

    tables = raw.get("tables")
    if not isinstance(tables, dict):
        raise ValueError(f"Invalid 'tables' type in {metadata_path}: expected dict")
    return tables
