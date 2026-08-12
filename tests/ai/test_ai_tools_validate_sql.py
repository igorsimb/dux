import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from langgraph.types import Command

from ai import ai_tools
from ai.ai_utils.logging_config import build_public_conversation_code, build_short_log_id
from ai.ai_utils import validate_sql as validate_sql_helpers
from ai.sql_dialect import SQLDialect


def _build_tool_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        config={"configurable": {"thread_id": "t1"}},
        state={},
        tool_call_id="tool-call-1",
    )


def _call_validate_sql(query: str):
    return getattr(ai_tools.validate_sql, "func")(query, _build_tool_runtime())


@pytest.fixture(autouse=True)
def _stub_stream_writer(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))


def test_validate_sql_rejects_clickhouse_placeholders_for_clickhouse(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("clickhouse_default", SQLDialect.CLICKHOUSE)],
    )

    result = _call_validate_sql("SELECT * FROM t WHERE region = {region:String}")

    assert isinstance(result, dict)
    assert result.get("error_code") == "UNBOUND_QUERY_PARAMETER"


def test_validate_sql_does_not_apply_clickhouse_placeholder_rule_for_other_dialects(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("postgres_default", SQLDialect.POSTGRES)],
    )
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_dialect_specific_query_guards",
        lambda query, dialect: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_uses_only_allowlisted_tables",
        lambda table_name_normalization_error: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_and_normalize_table_names",
        lambda query, dialect, allowed_full_tables, short_name_index: (
            None,
            "SELECT 1",
            [],
            None,
        ),
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {"analytics.customers"},
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {"customers": ["analytics.customers"]},
    )

    result = _call_validate_sql("SELECT * FROM t WHERE region = {region:String}")

    assert not (
        isinstance(result, dict)
        and result.get("error_code") == "UNBOUND_QUERY_PARAMETER"
    )


def test_validate_sql_rejects_disallowed_table(monkeypatch) -> None:
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("clickhouse_default", SQLDialect.CLICKHOUSE)],
    )
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_dialect_specific_query_guards",
        lambda query, dialect: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_uses_only_allowlisted_tables",
        lambda table_name_normalization_error: {
            "type": "reject",
            "error_code": "TABLE_NOT_ALLOWED",
            "reason": "Disallowed tables: unknown_table",
        },
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {"analytics.customers"},
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {"customers": ["analytics.customers"]},
    )

    result = _call_validate_sql("SELECT * FROM unknown_table")

    assert isinstance(result, dict)
    assert result.get("error_code") == "TABLE_NOT_ALLOWED"


def test_validate_sql_rejects_ambiguous_unqualified_table(monkeypatch) -> None:
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("clickhouse_default", SQLDialect.CLICKHOUSE)],
    )
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_dialect_specific_query_guards",
        lambda query, dialect: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_uses_only_allowlisted_tables",
        lambda table_name_normalization_error: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_and_normalize_table_names",
        lambda query, dialect, allowed_full_tables, short_name_index: (
            None,
            None,
            [],
            {
                "type": "reject",
                "error_code": "AMBIGUOUS_UNQUALIFIED_TABLE",
                "reason": "Ambiguous tables: customers. Use db.table format.",
            },
        ),
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {"analytics.customers", "archive.customers"},
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {"customers": ["analytics.customers", "archive.customers"]},
    )

    result = _call_validate_sql("SELECT * FROM customers")

    assert isinstance(result, dict)
    assert result.get("error_code") == "AMBIGUOUS_UNQUALIFIED_TABLE"


def test_validate_sql_reports_clickhouse_error_for_explicit_clickhouse_tables(monkeypatch) -> None:
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [
            ("clickhouse_default", SQLDialect.CLICKHOUSE),
            ("mssql_default", SQLDialect.MSSQL),
        ],
    )
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {
            "clickhouse_default": {
                "analytics.sales_fact",
                "analytics.customers",
            },
            "mssql_default": {"dbo.customer_orders"},
        }[source_id],
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {
            "clickhouse_default": {
                "sales_fact": ["analytics.sales_fact"],
                "customers": ["analytics.customers"],
            },
            "mssql_default": {"customer_orders": ["dbo.customer_orders"]},
        }[source_id],
    )

    result = _call_validate_sql(
        "SELECT count(DISTINCT sf.customer_id) AS [Customers] "
        "FROM analytics.sales_fact AS sf "
        "INNER JOIN analytics.customers AS c ON c.customer_id = sf.customer_id"
    )

    assert isinstance(result, dict)
    assert result.get("source_id") == "clickhouse_default"
    assert result.get("dialect") == "clickhouse"
    assert result.get("error_code") == "SQL_PARSE_ERROR"


def test_validate_sql_fails_fast_when_configured_dialect_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: (_ for _ in ()).throw(ValueError("bad dialect")),
    )

    with pytest.raises(ValueError, match="bad dialect"):
        _call_validate_sql("SELECT 1")


def test_validate_sql_rejects_non_readonly_query(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_is_read_only",
        lambda query: {
            "type": "reject",
            "error_code": "READ_ONLY_GUARD",
            "reason": "forbidden",
        },
    )

    result = _call_validate_sql("DELETE FROM analytics.customers")

    assert isinstance(result, dict)
    assert result == {
        "type": "reject",
        "error_code": "READ_ONLY_GUARD",
        "reason": "forbidden",
    }


def test_validate_sql_read_only_check_runs_before_dialect_loading(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_is_read_only",
        lambda query: {
            "type": "reject",
            "error_code": "READ_ONLY_GUARD",
            "reason": "forbidden",
        },
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: (_ for _ in ()).throw(ValueError("dialect must not be loaded")),
    )

    result = _call_validate_sql("DROP TABLE analytics.customers")

    assert isinstance(result, dict)
    assert result.get("error_code") == "READ_ONLY_GUARD"


def test_validate_sql_returns_parse_error_for_invalid_sql(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("postgres_default", SQLDialect.POSTGRES)],
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_dialect_specific_query_guards",
        lambda query, dialect: {
            "type": "reject",
            "error_code": "SQL_PARSE_ERROR",
            "reason": "Expected table name",
        },
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {"analytics.customers"},
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {"customers": ["analytics.customers"]},
    )

    result = _call_validate_sql("SELECT * FROM")

    assert isinstance(result, dict)
    assert result.get("error_code") == "SQL_PARSE_ERROR"


def test_validate_sql_success_updates_state_and_tool_message(monkeypatch) -> None:
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("postgres_default", SQLDialect.POSTGRES)],
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_dialect_specific_query_guards",
        lambda query, dialect: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_uses_only_allowlisted_tables",
        lambda table_name_normalization_error: None,
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {"analytics.customers"},
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {"customers": ["analytics.customers"]},
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_and_normalize_table_names",
        lambda query, dialect, allowed_full_tables, short_name_index: (
            None,
            "SELECT * FROM analytics.customers",
            ["analytics.customers"],
            None,
        ),
    )
    monkeypatch.setattr(
        ai_tools, "uuid4", lambda: UUID("00000000-0000-0000-0000-000000000123")
    )
    monkeypatch.setattr(ai_tools.time, "time", lambda: 1234.5)
    monkeypatch.setattr(
        ai_tools.logger,
        "debug",
        lambda message, *args: debug_calls.append((str(message), args)),
    )

    runtime = SimpleNamespace(
        config={"configurable": {"thread_id": "thread-42"}},
        state={"validated_queries": {"existing": {"status": "validated"}}},
        tool_call_id="tool-call-1",
    )

    result = getattr(ai_tools.validate_sql, "func")("SELECT * FROM customers", runtime)

    assert isinstance(result, Command)
    assert result.update is not None
    assert "existing" in result.update["validated_queries"]
    validated_id = "00000000-0000-0000-0000-000000000123"
    record = result.update["validated_queries"][validated_id]
    assert record["sql"] == "SELECT * FROM analytics.customers"
    assert record["source_id"] == "postgres_default"
    assert record["dialect"] == "postgres"
    assert record["tables"] == ["analytics.customers"]
    assert record["created_at"] == 1234.5
    assert record["expires_at"] == 1834.5
    assert record["thread_id"] == "thread-42"
    assert record["status"] == "validated"

    tool_message = result.update["messages"][0]
    assert tool_message.name == "validate_sql"
    payload = json.loads(tool_message.content)
    assert payload == {
        "type": "ok",
        "query": "SELECT * FROM analytics.customers",
        "original_query": "SELECT * FROM customers",
        "tables": ["analytics.customers"],
        "source_id": "postgres_default",
        "dialect": "postgres",
        "validated_id": validated_id,
        "validated_at": 1234.5,
    }
    log_text = "\n".join(" ".join([message, *(str(arg) for arg in args)]) for message, args in debug_calls)
    assert build_short_log_id("thread-42") in log_text
    assert "tool.sql" in log_text
    assert "validation_passed" in log_text
    assert all(
        build_public_conversation_code("thread-42") not in " ".join([message, *(str(arg) for arg in args)])
        for message, args in debug_calls
    )


def test_validate_sql_selects_single_successful_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [
            ("clickhouse_default", SQLDialect.CLICKHOUSE),
            ("mssql_default", SQLDialect.MSSQL),
        ],
    )

    def fake_parse(query, dialect):
        if dialect == SQLDialect.CLICKHOUSE:
            return {
                "type": "reject",
                "error_code": "SQL_PARSE_ERROR",
                "reason": "bad clickhouse syntax",
            }
        return None

    monkeypatch.setattr(
        validate_sql_helpers, "check_dialect_specific_query_guards", fake_parse
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_uses_only_allowlisted_tables",
        lambda error: None,
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {"dbo.sales_orders"},
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {"sales_orders": ["dbo.sales_orders"]},
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_and_normalize_table_names",
        lambda query, dialect, allowed_full_tables, short_name_index: (
            None,
            "SELECT * FROM dbo.sales_orders",
            ["dbo.sales_orders"],
            None,
        ),
    )
    monkeypatch.setattr(
        ai_tools, "uuid4", lambda: UUID("00000000-0000-0000-0000-000000000124")
    )
    monkeypatch.setattr(ai_tools.time, "time", lambda: 2000.0)

    result = getattr(ai_tools.validate_sql, "func")(
        "SELECT * FROM sales_orders", _build_tool_runtime()
    )

    assert isinstance(result, Command)
    assert result.update is not None
    payload = json.loads(result.update["messages"][0].content)
    assert payload["source_id"] == "mssql_default"
    assert payload["dialect"] == "tsql"


def test_validate_sql_rejects_when_multiple_candidates_succeed(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [
            ("clickhouse_default", SQLDialect.CLICKHOUSE),
            ("mssql_default", SQLDialect.MSSQL),
        ],
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_dialect_specific_query_guards",
        lambda query, dialect: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_uses_only_allowlisted_tables",
        lambda error: None,
    )
    monkeypatch.setattr(
        ai_tools, "load_allowed_tables_for_source", lambda source_id: {"dbo.shared"}
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {"shared": ["dbo.shared"]},
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_and_normalize_table_names",
        lambda query, dialect, allowed_full_tables, short_name_index: (
            None,
            "SELECT * FROM dbo.shared",
            ["dbo.shared"],
            None,
        ),
    )

    result = _call_validate_sql("SELECT * FROM shared")

    assert isinstance(result, dict)
    assert result.get("error_code") == "AMBIGUOUS_SOURCE_OR_DIALECT"


def test_validate_sql_rejects_missing_required_date_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("clickhouse_default", SQLDialect.CLICKHOUSE)],
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_dialect_specific_query_guards",
        lambda query, dialect: None,
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_uses_only_allowlisted_tables",
        lambda error: None,
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {"analytics.sales_fact"},
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {
            "sales_fact": ["analytics.sales_fact"]
        },
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_and_normalize_table_names",
        lambda query, dialect, allowed_full_tables, short_name_index: (
            object(),
            "SELECT * FROM analytics.sales_fact WHERE customer_id = 42",
            ["analytics.sales_fact"],
            None,
        ),
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_required_date_filter_for_tables",
        lambda parsed_ast, normalized_referenced_tables: {
            "type": "reject",
            "error_code": "MISSING_REQUIRED_DATE_FILTER",
            "reason": "Missing required date predicate on metadata-defined date column.",
            "tables_missing_date_filter": ["analytics.sales_fact"],
        },
    )

    result = _call_validate_sql(
        "SELECT * FROM analytics.sales_fact WHERE customer_id = 42"
    )

    assert isinstance(result, dict)
    assert result.get("error_code") == "MISSING_REQUIRED_DATE_FILTER"


def test_validate_sql_routes_query_to_clickhouse_when_only_clickhouse_candidate_passes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ai_tools, "uuid4", lambda: UUID("00000000-0000-0000-0000-000000000126")
    )
    monkeypatch.setattr(ai_tools.time, "time", lambda: 4000.0)
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [
            ("clickhouse_default", SQLDialect.CLICKHOUSE),
            ("mssql_default", SQLDialect.MSSQL),
        ],
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {
            "clickhouse_default": {"analytics.sales_fact"},
            "mssql_default": {"dbo.customer_orders"},
        }[source_id],
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {
            "clickhouse_default": {"sales_fact": ["analytics.sales_fact"]},
            "mssql_default": {"customer_orders": ["dbo.customer_orders"]},
        }[source_id],
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_required_date_filter_for_tables",
        lambda canonical_query, referenced_tables: None,
    )

    result = _call_validate_sql(
        "SELECT region, sum(revenue) AS total_revenue "
        "FROM sales_fact "
        "WHERE event_date >= today() - 7 "
        "GROUP BY region "
        "ORDER BY total_revenue DESC "
        "LIMIT 10"
    )

    assert isinstance(result, Command)
    assert result.update is not None
    payload = json.loads(result.update["messages"][0].content)
    assert payload["source_id"] == "clickhouse_default"
    assert payload["dialect"] == "clickhouse"
    assert payload["tables"] == ["analytics.sales_fact"]
    assert "analytics.sales_fact" in payload["query"].lower()


def test_validate_sql_rejects_query_when_tables_are_disallowed_for_all_candidates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [
            ("clickhouse_default", SQLDialect.CLICKHOUSE),
            ("mssql_default", SQLDialect.MSSQL),
        ],
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_tables_for_source",
        lambda source_id: {
            "clickhouse_default": {"analytics.sales_fact"},
            "mssql_default": {"dbo.customer_orders"},
        }[source_id],
    )
    monkeypatch.setattr(
        ai_tools,
        "get_short_name_index_for_source",
        lambda source_id: {
            "clickhouse_default": {"sales_fact": ["analytics.sales_fact"]},
            "mssql_default": {"customer_orders": ["dbo.customer_orders"]},
        }[source_id],
    )

    result = _call_validate_sql(
        "SELECT o.order_id, sf.order_id "
        "FROM dbo.customer_orders AS o "
        "INNER JOIN analytics.sales_fact AS sf ON sf.order_id = o.order_id"
    )

    assert isinstance(result, dict)
    assert result.get("error_code") == "TABLE_NOT_ALLOWED"
    assert "Disallowed tables:" in str(result.get("reason"))
    assert any(
        table in str(result.get("reason"))
        for table in ["dbo.customer_orders", "analytics.sales_fact"]
    )
    assert set(result.get("allowed_tables", [])) in [
        {"analytics.sales_fact"},
        {"dbo.customer_orders"},
    ]
