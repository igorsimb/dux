import json
from pathlib import Path

import pytest

from ai.sql_dialect import SQLDialect
from core.db_config import source_registry


def _write_sql_sources_config(tmp_path: Path, config: dict[str, object]) -> Path:
    config_path = tmp_path / "sql_sources.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_load_sql_sources_parses_valid_config(tmp_path: Path) -> None:
    config_path = _write_sql_sources_config(
        tmp_path,
        {
            "sources": [
                {
                    "id": "clickhouse_default",
                    "dialect": "clickhouse",
                    "driver": "clickhouse",
                    "env": {
                        "host": "CLICKHOUSE_HOST",
                        "port": "CLICKHOUSE_PORT",
                    },
                },
                {
                    "id": "mssql_default",
                    "dialect": "tsql",
                    "driver": "mssql",
                    "env": {
                        "host": "MSSQL_HOST",
                        "database": "MSSQL_DATABASE",
                    },
                },
            ]
        },
    )

    sources = source_registry.load_sql_sources(config_path)

    assert len(sources) == 2
    assert sources[0].id == "clickhouse_default"
    assert sources[0].dialect is SQLDialect.CLICKHOUSE
    assert sources[1].id == "mssql_default"
    assert sources[1].dialect is SQLDialect.MSSQL


def test_load_sql_sources_requires_sources_key(tmp_path: Path) -> None:
    config_path = _write_sql_sources_config(tmp_path, {})

    with pytest.raises(ValueError) as exc_info:
        source_registry.load_sql_sources(config_path)

    assert "Missing required key 'sources'" in str(exc_info.value)


def test_load_sql_sources_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    config_path = _write_sql_sources_config(
        tmp_path,
        {
            "sources": [
                {
                    "id": "mssql_default",
                    "dialect": "tsql",
                    "driver": "mssql",
                    "env": {"host": "MSSQL_HOST"},
                },
                {
                    "id": "mssql_default",
                    "dialect": "tsql",
                    "driver": "mssql",
                    "env": {"host": "MSSQL_HOST"},
                },
            ]
        },
    )

    with pytest.raises(ValueError) as exc_info:
        source_registry.load_sql_sources(config_path)

    assert "Duplicate source id" in str(exc_info.value)


def test_get_source_by_id_returns_matching_source(tmp_path: Path) -> None:
    config_path = _write_sql_sources_config(
        tmp_path,
        {
            "sources": [
                {
                    "id": "mssql_default",
                    "dialect": "tsql",
                    "driver": "mssql",
                    "env": {"host": "MSSQL_HOST"},
                }
            ]
        },
    )

    source = source_registry.get_source_by_id("mssql_default", config_path)

    assert source.id == "mssql_default"
    assert source.dialect is SQLDialect.MSSQL


def test_get_sources_by_dialect_filters_sources(tmp_path: Path) -> None:
    config_path = _write_sql_sources_config(
        tmp_path,
        {
            "sources": [
                {
                    "id": "clickhouse_default",
                    "dialect": "clickhouse",
                    "driver": "clickhouse",
                    "env": {"host": "CLICKHOUSE_HOST"},
                },
                {
                    "id": "mssql_default",
                    "dialect": "tsql",
                    "driver": "mssql",
                    "env": {"host": "MSSQL_HOST"},
                },
            ]
        },
    )

    sources = source_registry.get_sources_by_dialect(SQLDialect.MSSQL, config_path)

    assert [item.id for item in sources] == ["mssql_default"]
