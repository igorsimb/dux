import pytest
import sqlglot

from ai.sql_dialect import SQLDialect
from tests.ai.fixtures import supported_dialects


def test_sql_dialect_values(supported_dialects: tuple[str, ...]) -> None:
    assert tuple(d.value for d in SQLDialect) == supported_dialects


def test_from_raw_normalizes_value() -> None:
    assert SQLDialect.from_raw(" ClickHouse ") is SQLDialect.CLICKHOUSE


def test_from_raw_returns_existing_dialect() -> None:
    assert SQLDialect.from_raw(SQLDialect.MSSQL) is SQLDialect.MSSQL


def test_from_raw_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError) as exc_info:
        SQLDialect.from_raw("oracle")

    message = str(exc_info.value)
    assert "Unsupported SQL dialect" in message
    assert "oracle" in message
    assert "clickhouse" in message


def test_sqlglot_parse_accepts_enum() -> None:
    expression = sqlglot.parse_one("SELECT 1", read=SQLDialect.CLICKHOUSE)

    assert expression.sql(dialect=SQLDialect.CLICKHOUSE) == "SELECT 1"
