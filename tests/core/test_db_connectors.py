from __future__ import annotations

import pytest

from ai.sql_dialect import SQLDialect
from core.db_config import clickhouse_connector, mssql_connector, sqlite_connector
from core.db_config.source_registry import SQLSourceConfig


def test_create_clickhouse_sql_database_uses_required_env(
    monkeypatch,
) -> None:
    source = SQLSourceConfig(
        id="clickhouse_default",
        dialect=SQLDialect.CLICKHOUSE,
        driver="clickhouse",
        env={
            "host": "CLICKHOUSE_HOST",
            "port": "CLICKHOUSE_PORT",
            "user": "CLICKHOUSE_USER",
            "password": "CLICKHOUSE_PASSWORD",
        },
        default_database="analytics",
    )
    monkeypatch.setenv("CLICKHOUSE_HOST", "127.0.0.1")
    monkeypatch.setenv("CLICKHOUSE_PORT", "8123")
    monkeypatch.setenv("CLICKHOUSE_USER", "user1")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "pass1")

    captured: dict[str, object] = {}

    def fake_create_engine(url):
        captured["url"] = url
        return "engine"

    def fake_sql_database(engine, include_tables, sample_rows_in_table_info):
        captured["engine"] = engine
        captured["include_tables"] = include_tables
        captured["sample_rows_in_table_info"] = sample_rows_in_table_info
        return "db"

    monkeypatch.setattr(clickhouse_connector, "create_engine", fake_create_engine)
    monkeypatch.setattr(clickhouse_connector, "SQLDatabase", fake_sql_database)

    db = clickhouse_connector.create_clickhouse_sql_database(source, ["customers"])

    assert db == "db"
    assert captured["engine"] == "engine"
    assert captured["include_tables"] == ["customers"]
    assert captured["sample_rows_in_table_info"] == 0
    assert str(captured["url"]).startswith(
        "clickhouse+http://user1:***@127.0.0.1:8123/analytics"
    )


def test_create_clickhouse_sql_database_raises_when_port_env_is_missing(
    monkeypatch,
) -> None:
    source = SQLSourceConfig(
        id="clickhouse_default",
        dialect=SQLDialect.CLICKHOUSE,
        driver="clickhouse",
        env={
            "host": "CLICKHOUSE_HOST",
            "port": "CLICKHOUSE_PORT",
            "user": "CLICKHOUSE_USER",
            "password": "CLICKHOUSE_PASSWORD",
        },
        default_database="analytics",
    )
    monkeypatch.setenv("CLICKHOUSE_HOST", "127.0.0.1")
    monkeypatch.delenv("CLICKHOUSE_PORT", raising=False)
    monkeypatch.setenv("CLICKHOUSE_USER", "user1")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "pass1")

    with pytest.raises(ValueError) as exc_info:
        clickhouse_connector.create_clickhouse_sql_database(source, ["customers"])

    assert "CLICKHOUSE_PORT" in str(exc_info.value)


def test_create_mssql_sql_database_uses_required_env(
    monkeypatch,
) -> None:
    source = SQLSourceConfig(
        id="mssql_default",
        dialect=SQLDialect.MSSQL,
        driver="mssql",
        env={
            "host": "MSSQL_HOST",
            "port": "MSSQL_PORT",
            "user": "MSSQL_USER",
            "password": "MSSQL_PASSWORD",
            "database": "MSSQL_DATABASE",
        },
    )
    monkeypatch.setenv("MSSQL_HOST", "10.1.2.3")
    monkeypatch.setenv("MSSQL_PORT", "1433")
    monkeypatch.setenv("MSSQL_USER", "sa")
    monkeypatch.setenv("MSSQL_PASSWORD", "secret")
    monkeypatch.setenv("MSSQL_DATABASE", "erp")
    monkeypatch.setenv("MSSQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")

    captured: dict[str, object] = {}

    def fake_create_engine(url):
        captured["url"] = url
        return "engine"

    def fake_sql_database(engine, include_tables, sample_rows_in_table_info):
        captured["engine"] = engine
        captured["include_tables"] = include_tables
        captured["sample_rows_in_table_info"] = sample_rows_in_table_info
        return "db"

    monkeypatch.setattr(mssql_connector, "create_engine", fake_create_engine)
    monkeypatch.setattr(mssql_connector, "SQLDatabase", fake_sql_database)

    db = mssql_connector.create_mssql_sql_database(source, ["sales_orders"])

    assert db == "db"
    assert captured["engine"] == "engine"
    assert captured["include_tables"] == ["sales_orders"]
    assert captured["sample_rows_in_table_info"] == 0
    assert str(captured["url"]).startswith("mssql+pyodbc://sa:***@10.1.2.3:1433/erp")


def test_create_mssql_sql_database_raises_when_user_env_is_missing(
    monkeypatch,
) -> None:
    source = SQLSourceConfig(
        id="mssql_default",
        dialect=SQLDialect.MSSQL,
        driver="mssql",
        env={
            "host": "MSSQL_HOST",
            "port": "MSSQL_PORT",
            "user": "MSSQL_USER",
            "password": "MSSQL_PASSWORD",
            "database": "MSSQL_DATABASE",
        },
    )
    monkeypatch.setenv("MSSQL_HOST", "10.1.2.3")
    monkeypatch.setenv("MSSQL_PORT", "1433")
    monkeypatch.delenv("MSSQL_USER", raising=False)
    monkeypatch.setenv("MSSQL_PASSWORD", "secret")
    monkeypatch.setenv("MSSQL_DATABASE", "erp")

    with pytest.raises(ValueError) as exc_info:
        mssql_connector.create_mssql_sql_database(source, ["sales_orders"])

    assert "MSSQL_USER" in str(exc_info.value)


def _sqlite_source() -> SQLSourceConfig:
    return SQLSourceConfig(
        id="chinook",
        dialect=SQLDialect.SQLITE,
        driver="sqlite",
        env={"path": "CHINOOK_DATABASE_PATH"},
        default_database="Chinook.db",
    )


def test_create_sqlite_sql_database_uses_environment_override(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "custom.db"
    database_path.touch()
    monkeypatch.setenv("CHINOOK_DATABASE_PATH", str(database_path))

    captured: dict[str, object] = {}

    def fake_create_engine(url):
        captured["url"] = url
        return "engine"

    def fake_sql_database(engine, include_tables, sample_rows_in_table_info):
        captured["engine"] = engine
        captured["include_tables"] = include_tables
        captured["sample_rows_in_table_info"] = sample_rows_in_table_info
        return "db"

    monkeypatch.setattr(sqlite_connector, "create_engine", fake_create_engine)
    monkeypatch.setattr(sqlite_connector, "SQLDatabase", fake_sql_database)

    db = sqlite_connector.create_sqlite_sql_database(_sqlite_source(), ["Invoice"])

    assert db == "db"
    assert captured["engine"] == "engine"
    assert captured["include_tables"] == ["Invoice"]
    assert captured["sample_rows_in_table_info"] == 0
    assert captured["url"].database == str(database_path.resolve())


def test_resolve_sqlite_database_path_uses_repository_relative_fallback(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "Chinook.db"
    database_path.touch()
    monkeypatch.delenv("CHINOOK_DATABASE_PATH", raising=False)
    monkeypatch.setattr(sqlite_connector, "PROJECT_ROOT", tmp_path)

    resolved = sqlite_connector.resolve_sqlite_database_path(_sqlite_source())

    assert resolved == database_path.resolve()


def test_resolve_sqlite_database_path_resolves_relative_override_from_project_root(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "data" / "Chinook.db"
    database_path.parent.mkdir()
    database_path.touch()
    monkeypatch.setenv("CHINOOK_DATABASE_PATH", "data/Chinook.db")
    monkeypatch.setattr(sqlite_connector, "PROJECT_ROOT", tmp_path)

    resolved = sqlite_connector.resolve_sqlite_database_path(_sqlite_source())

    assert resolved == database_path.resolve()


def test_resolve_sqlite_database_path_rejects_missing_file(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CHINOOK_DATABASE_PATH", "missing.db")
    monkeypatch.setattr(sqlite_connector, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError) as exc_info:
        sqlite_connector.resolve_sqlite_database_path(_sqlite_source())

    assert "chinook" in str(exc_info.value)
    assert "missing.db" in str(exc_info.value)


def test_resolve_sqlite_database_path_rejects_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CHINOOK_DATABASE_PATH", str(tmp_path))

    with pytest.raises(ValueError) as exc_info:
        sqlite_connector.resolve_sqlite_database_path(_sqlite_source())

    assert "is not a file" in str(exc_info.value)
