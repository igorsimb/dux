import json
from pathlib import Path

import pytest

from ai.sql_dialect import SQLDialect
import core.db_config.source_database_router as router
from core.db_config.source_registry import SQLSourceConfig


def test_get_include_tables_for_source_filters_and_deduplicates(
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
                        "table": "analytics.customers",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": False,
                    },
                    {
                        "table": "dbo.customers",
                        "source": "mssql_default",
                        "sql_dialect": "tsql",
                        "allowed": True,
                    },
                    {
                        "table": "analytics.should_not_be_visible",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": False,
                    },
                    {
                        "table": "catalog.products",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(router, "TABLE_DESCRIPTIONS_PATH", descriptions_path)

    include_tables = router.get_include_tables_for_source("clickhouse_default")

    assert include_tables == ["customers", "products"]


def test_get_include_tables_for_source_filters_by_database_name(
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
                        "table": "catalog.products",
                        "source": "clickhouse_default",
                        "sql_dialect": "clickhouse",
                        "allowed": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(router, "TABLE_DESCRIPTIONS_PATH", descriptions_path)

    include_tables = router.get_include_tables_for_source(
        "clickhouse_default", database_name="analytics"
    )

    assert include_tables == ["customers"]


def test_get_sql_database_for_source_routes_clickhouse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clickhouse_source = SQLSourceConfig(
        id="clickhouse_default",
        dialect=SQLDialect.CLICKHOUSE,
        driver="clickhouse",
        env={"host": "CLICKHOUSE_HOST"},
        default_database="analytics",
    )

    monkeypatch.setattr(router, "get_source_by_id", lambda source_id: clickhouse_source)
    monkeypatch.setattr(
        router,
        "get_include_tables_for_source",
        lambda source_id, path=None, database_name=None: ["customers"],
    )

    captured: dict[str, object] = {}

    def fake_clickhouse_builder(source: SQLSourceConfig, include_tables: list[str]):
        captured["source"] = source
        captured["include_tables"] = include_tables
        return "clickhouse_db"

    monkeypatch.setattr(
        router, "create_clickhouse_sql_database", fake_clickhouse_builder
    )

    sql_database = router.get_sql_database_for_source("clickhouse_default")

    assert sql_database == "clickhouse_db"
    assert captured["source"] == clickhouse_source
    assert captured["include_tables"] == ["customers"]


def test_get_sql_database_for_source_routes_mssql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mssql_source = SQLSourceConfig(
        id="mssql_default",
        dialect=SQLDialect.MSSQL,
        driver="mssql",
        env={"host": "MSSQL_HOST", "database": "MSSQL_DATABASE"},
    )

    monkeypatch.setattr(router, "get_source_by_id", lambda source_id: mssql_source)
    monkeypatch.setattr(
        router,
        "get_include_tables_for_source",
        lambda source_id, path=None, database_name=None: ["sales_orders"],
    )
    monkeypatch.setattr(
        router, "create_mssql_sql_database", lambda source, include_tables: "mssql_db"
    )

    sql_database = router.get_sql_database_for_source("mssql_default")

    assert sql_database == "mssql_db"


def test_get_sql_database_for_source_routes_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_source = SQLSourceConfig(
        id="chinook",
        dialect=SQLDialect.SQLITE,
        driver="sqlite",
        env={"path": "CHINOOK_DATABASE_PATH"},
        default_database="Chinook.db",
    )

    monkeypatch.setattr(router, "get_source_by_id", lambda source_id: sqlite_source)
    monkeypatch.setattr(
        router,
        "get_include_tables_for_source",
        lambda source_id, path=None, database_name=None: ["Invoice", "InvoiceLine"],
    )
    monkeypatch.setattr(
        router, "create_sqlite_sql_database", lambda source, include_tables: "sqlite_db"
    )

    sql_database = router.get_sql_database_for_source("chinook")

    assert sql_database == "sqlite_db"


def test_get_sql_database_for_source_rejects_unknown_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown_source = SQLSourceConfig(
        id="legacy_source",
        dialect=SQLDialect.MSSQL,
        driver="legacydb",
        env={"host": "LEGACY_HOST"},
    )

    monkeypatch.setattr(router, "get_source_by_id", lambda source_id: unknown_source)
    monkeypatch.setattr(
        router,
        "get_include_tables_for_source",
        lambda source_id, path=None, database_name=None: ["legacy_table"],
    )

    with pytest.raises(ValueError) as exc_info:
        router.get_sql_database_for_source("legacy_source")

    assert "Unsupported source driver" in str(exc_info.value)
