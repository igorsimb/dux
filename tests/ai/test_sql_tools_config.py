import json
from pathlib import Path

import pytest
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

import ai.ai_utils.sql_tools as sql_tools
from ai.sql_dialect import SQLDialect
from ai.sql_guard import GuardedQuerySQLDatabaseTool


def test_build_guarded_query_tool_returns_backend_execution_tool() -> None:
    db = SQLDatabase(create_engine("sqlite:///:memory:"))

    tool = sql_tools.build_guarded_query_tool(db)

    assert isinstance(tool, GuardedQuerySQLDatabaseTool)
    assert tool.db is db
    assert tool.name == "sql_db_query"


def test_load_dialect_and_tables_accepts_list_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "table_descriptions.json"
    config_path.write_text(
        json.dumps(
            {
                "sql_dialect": "clickhouse",
                "tables": [{"table": "analytics.customers", "allowed": True}],
            }
        ),
        encoding="utf-8",
    )

    dialect, tables = sql_tools.load_dialect_and_tables(
        config_path, expected_tables_type=list
    )

    assert dialect is SQLDialect.CLICKHOUSE
    assert isinstance(tables, list)
    assert tables[0]["table"] == "analytics.customers"


def test_load_dialect_and_tables_requires_sql_dialect(tmp_path: Path) -> None:
    config_path = tmp_path / "table_descriptions.json"
    config_path.write_text(json.dumps({"tables": []}), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        sql_tools.load_dialect_and_tables(config_path, expected_tables_type=list)

    assert "Missing required key 'sql_dialect'" in str(exc_info.value)


def test_load_dialect_and_tables_rejects_unsupported_dialect(tmp_path: Path) -> None:
    config_path = tmp_path / "table_descriptions.json"
    config_path.write_text(
        json.dumps({"sql_dialect": "oracle", "tables": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError) as exc_info:
        sql_tools.load_dialect_and_tables(config_path, expected_tables_type=list)

    assert "Unsupported SQL dialect" in str(exc_info.value)


def test_configured_sql_dialect_raises_on_multiple_allowlisted_dialects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    },
                    {
                        "table": "analytics.product_catalog",
                        "source": "mssql_default",
                        "sql_dialect": "tsql",
                        "allowed": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))

    sql_tools.get_configured_sql_dialect.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_table_metadata.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()
    sql_tools.get_short_name_index.cache_clear()

    with pytest.raises(ValueError) as exc_info:
        sql_tools.get_configured_sql_dialect()

    assert "Multiple SQL dialects configured in allowlisted tables" in str(
        exc_info.value
    )


def test_load_table_descriptions_requires_source_and_sql_dialect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "allowed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))

    sql_tools.get_configured_sql_dialect.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()
    sql_tools.get_short_name_index.cache_clear()

    with pytest.raises(ValueError) as exc_info:
        sql_tools.load_table_descriptions()

    assert "Missing required field 'source'" in str(exc_info.value)


def test_load_table_descriptions_accepts_row_level_sql_dialect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    },
                    {
                        "table": "analytics.hidden_table",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))

    sql_tools.get_configured_sql_dialect.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()
    sql_tools.get_short_name_index.cache_clear()

    descriptions = sql_tools.load_table_descriptions()
    allowed_tables = sql_tools.load_allowed_tables()

    assert len(descriptions) == 1
    assert descriptions[0]["table"] == "analytics.customers"
    assert descriptions[0]["source"] == "clickhouse_default"
    assert descriptions[0]["sql_dialect"] == "clickhouse"
    assert allowed_tables == {"analytics.customers"}


def test_load_table_descriptions_rejects_unknown_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "source": "unknown_source",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))

    sql_tools.get_configured_sql_dialect.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()
    sql_tools.get_short_name_index.cache_clear()

    with pytest.raises(ValueError) as exc_info:
        sql_tools.load_table_descriptions()

    assert "Unknown source" in str(exc_info.value)


def test_load_table_descriptions_rejects_source_dialect_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "source": "clickhouse_default",
                        "sql_dialect": "tsql",
                        "allowed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))

    sql_tools.get_configured_sql_dialect.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()
    sql_tools.get_short_name_index.cache_clear()

    with pytest.raises(ValueError) as exc_info:
        sql_tools.load_table_descriptions()

    assert "Dialect mismatch for table" in str(exc_info.value)


def test_get_configured_source_id_returns_single_allowlisted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    },
                    {
                        "table": "analytics.hidden_table",
                        "source": "mssql_default",
                        "sql_dialect": "tsql",
                        "allowed": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))

    sql_tools.get_configured_source_id.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()

    source_id = sql_tools.get_configured_source_id()

    assert source_id == "clickhouse_default"


def test_get_configured_source_id_raises_on_multiple_allowlisted_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    },
                    {
                        "table": "dbo.sales_orders",
                        "source": "mssql_default",
                        "sql_dialect": "tsql",
                        "allowed": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))

    sql_tools.get_configured_source_id.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()

    with pytest.raises(ValueError) as exc_info:
        sql_tools.get_configured_source_id()

    assert "Multiple sources configured in allowlisted tables" in str(exc_info.value)


def test_load_table_metadata_excludes_disallowed_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptions_path = tmp_path / "table_descriptions.json"
    metadata_path = tmp_path / "table_metadata.json"

    descriptions_path.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "table": "analytics.customers",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    },
                    {
                        "table": "analytics.hidden_table",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "tables": {
                    "analytics.customers": {
                        "table": "analytics.customers",
                        "description": "allowed",
                    },
                    "analytics.hidden_table": {
                        "table": "analytics.hidden_table",
                        "description": "disallowed",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sql_tools, "TABLE_DESCRIPTIONS_PATH", str(descriptions_path))
    monkeypatch.setattr(sql_tools, "TABLE_METADATA_PATH", str(metadata_path))

    sql_tools.get_configured_sql_dialect.cache_clear()
    sql_tools.get_configured_source_id.cache_clear()
    sql_tools.load_table_descriptions.cache_clear()
    sql_tools.load_table_metadata.cache_clear()
    sql_tools.load_allowed_tables.cache_clear()
    sql_tools.load_allowed_source_candidates.cache_clear()
    sql_tools.load_allowed_tables_for_source.cache_clear()
    sql_tools.get_short_name_index.cache_clear()
    sql_tools.get_short_name_index_for_source.cache_clear()

    metadata = sql_tools.load_table_metadata()

    assert set(metadata.keys()) == {"analytics.customers"}
    assert metadata["analytics.customers"]["description"] == "allowed"


def test_get_run_query_tool_for_source_builds_registered_factory_once() -> None:
    sql_tools.clear_run_query_tools()
    calls: list[str] = []

    sql_tools.set_run_query_tool_factory_for_source(
        "mssql_default", lambda: calls.append("built") or "tool:mssql"
    )

    first_tool = sql_tools.get_run_query_tool_for_source("mssql_default")
    second_tool = sql_tools.get_run_query_tool_for_source("mssql_default")

    assert first_tool == "tool:mssql"
    assert second_tool == "tool:mssql"
    assert calls == ["built"]
