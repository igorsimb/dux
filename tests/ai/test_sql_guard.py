import sqlglot

from ai.sql_dialect import SQLDialect
from ai.sql_guard import GuardedQuerySQLDatabaseTool, resolve_sql_tables_to_allowed_names_with_ast


def test_resolve_sql_tables_uses_passed_dialect(monkeypatch):
    captured_reads: list[SQLDialect] = []
    real_parse_one = sqlglot.parse_one

    def fake_parse_one(query: str, read: SQLDialect):
        captured_reads.append(read)
        return real_parse_one(query, read=SQLDialect.CLICKHOUSE)

    monkeypatch.setattr("ai.sql_guard.sqlglot.parse_one", fake_parse_one)

    _, canonical_sql, normalized_tables, error = (
        resolve_sql_tables_to_allowed_names_with_ast(
            query="SELECT * FROM analytics.customers",
            allowed_full_tables={"analytics.customers"},
            short_name_index={"customers": ["analytics.customers"]},
            dialect=SQLDialect.POSTGRES,
        )
    )

    assert captured_reads == [SQLDialect.POSTGRES]
    assert error is None
    assert canonical_sql is not None
    assert normalized_tables == ["analytics.customers"]


def test_resolve_sql_tables_returns_parse_error_for_invalid_sql_in_any_dialect() -> (
    None
):
    _, canonical_sql, normalized_tables, error = (
        resolve_sql_tables_to_allowed_names_with_ast(
            query="SELECT * FROM",
            allowed_full_tables={"analytics.customers"},
            short_name_index={"customers": ["analytics.customers"]},
            dialect=SQLDialect.POSTGRES,
        )
    )

    assert canonical_sql is None
    assert normalized_tables == []
    assert isinstance(error, dict)
    assert error.get("error_code") == "SQL_PARSE_ERROR"


def test_guarded_query_tool_run_structured_returns_row_dicts() -> None:
    class Db:
        def _execute(self, query: str, fetch: str = "all"):
            assert query == "SELECT 1 AS answer"
            assert fetch == "all"
            return [{"answer": 1}]

    tool = GuardedQuerySQLDatabaseTool.model_construct(db=Db())

    assert tool.run_structured("SELECT 1 AS answer") == [{"answer": 1}]


def test_guarded_query_tool_run_structured_reuses_readonly_guard() -> None:
    class Db:
        def _execute(self, query: str, fetch: str = "all"):
            raise AssertionError("unsafe SQL should not execute")

    tool = GuardedQuerySQLDatabaseTool.model_construct(db=Db())

    assert tool.run_structured("DROP TABLE customers").startswith("Error: forbidden SQL")
