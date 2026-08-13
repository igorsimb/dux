"""Route source ids to SQLDatabase connectors and per-source table subsets."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.utilities import SQLDatabase

from core.db_config.clickhouse_connector import create_clickhouse_sql_database
from core.db_config.mssql_connector import create_mssql_sql_database
from core.db_config.sqlite_connector import create_sqlite_sql_database
from core.db_config.source_registry import get_source_by_id

TABLE_DESCRIPTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "ai" / "table_descriptions.json"
)


def _split_table_reference(table_name: str) -> tuple[str | None, str]:
    table_database, separator, short_name = table_name.partition(".")
    if not separator:
        return None, table_database
    return table_database, short_name


def get_include_tables_for_source(
    source_id: str,
    path: str | Path | None = None,
    database_name: str | None = None,
) -> list[str]:
    """Return de-duplicated short table names for one source id.

    When `database_name` is provided, only fully-qualified tables in that database are included.
    """
    rows = load_table_description_rows(path)
    include_tables: list[str] = []
    seen: set[str] = set()
    database_name_normalized = (database_name or "").strip().lower()

    for row in rows:
        if row["source"] != source_id:
            continue
        table_name = row["table"]
        table_database, short_name = _split_table_reference(table_name)
        if database_name_normalized and table_database is not None:
            if table_database.strip().lower() != database_name_normalized:
                continue
        if short_name in seen:
            continue
        seen.add(short_name)
        include_tables.append(short_name)

    return include_tables


def get_sql_database_for_source(source_id: str) -> SQLDatabase:
    """Create SQLDatabase for one configured source id."""
    source = get_source_by_id(source_id)
    include_tables = get_include_tables_for_source(
        source_id,
        database_name=source.default_database
        if source.driver == "clickhouse"
        else None,
    )

    if source.driver == "clickhouse":
        return create_clickhouse_sql_database(source, include_tables)
    if source.driver == "mssql":
        return create_mssql_sql_database(source, include_tables)
    if source.driver == "sqlite":
        return create_sqlite_sql_database(source, include_tables)
    raise ValueError(
        f"Unsupported source driver '{source.driver}' for source '{source.id}'"
    )


def load_table_description_rows(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load and minimally validate table description rows for source routing."""
    descriptions_path = Path(path) if path is not None else TABLE_DESCRIPTIONS_PATH
    raw = json.loads(descriptions_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON root in {descriptions_path}: expected object")

    rows = raw.get("tables")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid 'tables' type in {descriptions_path}: expected list")

    normalized_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid table row at index {index} in {descriptions_path}: expected object"
            )

        table_name = str(row.get("table", "")).strip()
        if not table_name:
            raise ValueError(
                f"Missing required field 'table' at index {index} in {descriptions_path}"
            )
        source_id = str(row.get("source", "")).strip()
        if not source_id:
            raise ValueError(
                f"Missing required field 'source' for table '{table_name}' in {descriptions_path}"
            )
        if not bool(row.get("allowed")):
            continue

        normalized_rows.append({"table": table_name, "source": source_id})

    return normalized_rows
