from types import SimpleNamespace
from uuid import UUID

from langchain.messages import ToolMessage
from langgraph.types import Command

from ai import ai_tools
from ai.ai_utils.logging_config import build_public_conversation_code, build_short_log_id
from ai.sql_dialect import SQLDialect


def runtime_with_state(state: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        config={"configurable": {"thread_id": "t1"}},
        state=state or {},
        tool_call_id="tool-call-1",
    )


def test_get_sql_result_row_limit_uses_explicit_limit() -> None:
    assert ai_tools._get_explicit_query_row_limit("SELECT * FROM offers LIMIT 75", "clickhouse") == 75


def test_get_sql_result_row_limit_uses_explicit_tsql_top() -> None:
    assert ai_tools._get_explicit_query_row_limit("SELECT TOP 75 * FROM dbo.customer_orders", "tsql") == 75


def test_get_sql_result_row_limit_falls_back_to_default_limit() -> None:
    assert ai_tools._get_explicit_query_row_limit("SELECT * FROM offers", "clickhouse") is None


def test_execute_validated_sql_dispatches_to_source_specific_tool(monkeypatch) -> None:
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        ai_tools.logger,
        "debug",
        lambda message, *args: debug_calls.append((str(message), args)),
    )

    captured: dict[str, object] = {}

    class Tool:
        def run_structured(self, query):
            captured["query"] = query
            return [{"answer": 1}]

    def get_tool_for_source(source_id: str):
        captured["source_id"] = source_id
        return Tool()

    monkeypatch.setattr(ai_tools, "get_run_query_tool_for_source", get_tool_for_source)

    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT 1",
                    "source_id": "mssql_default",
                    "dialect": "tsql",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, Command)
    assert captured["source_id"] == "mssql_default"
    assert captured["query"] == "SELECT 1"
    log_text = "\n".join(" ".join([message, *(str(arg) for arg in args)]) for message, args in debug_calls)
    assert build_short_log_id("t1") in log_text
    assert "tool.sql" in log_text
    assert all(
        build_public_conversation_code("t1") not in " ".join([message, *(str(arg) for arg in args)])
        for message, args in debug_calls
    )


def test_execute_validated_sql_rejects_when_source_tool_missing(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        ai_tools, "get_run_query_tool_for_source", lambda source_id: None
    )

    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT 1",
                    "source_id": "clickhouse_default",
                    "dialect": "clickhouse",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, dict)
    assert result.get("error_code") == "SQL_QUERY_TOOL_NOT_CONFIGURED"
    assert "clickhouse_default" in str(result.get("reason"))


def test_execute_validated_sql_rejects_when_source_tool_init_fails(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    def raise_tool_init_error(source_id: str):
        raise RuntimeError(f"cannot connect to {source_id}")

    monkeypatch.setattr(
        ai_tools, "get_run_query_tool_for_source", raise_tool_init_error
    )

    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT 1",
                    "source_id": "mssql_default",
                    "dialect": "tsql",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, dict)
    assert result.get("error_code") == "SQL_QUERY_TOOL_INIT_ERROR"
    assert "mssql_default" in str(result.get("reason"))


def test_execute_validated_sql_appends_data_table_block_from_structured_rows(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    class Tool:
        def run_structured(self, query):
            return [
                {"customer": "A", "qty": 10, "price": 12.5, "active": True, "updated_at": None},
                {"customer": "B", "qty": 20, "price": 14.0, "active": False, "updated_at": None},
            ]

    monkeypatch.setattr(ai_tools, "get_run_query_tool_for_source", lambda source_id: Tool())
    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT customer, qty FROM offers",
                    "source_id": "mssql_default",
                    "dialect": "tsql",
                    "tables": ["dbo.offers"],
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            },
            "sql_result_table_blocks": [{"id": "existing", "type": "data_table"}],
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, Command)
    assert result.update is not None
    blocks = result.update["sql_result_table_blocks"]
    assert blocks[0] == {"id": "existing", "type": "data_table"}
    block = blocks[1]
    assert block["type"] == "data_table"
    assert block["columns"] == [
        {"key": "customer", "label": "customer", "type": "string"},
        {"key": "qty", "label": "qty", "type": "number"},
        {"key": "price", "label": "price", "type": "number"},
        {"key": "active", "label": "active", "type": "boolean"},
        {"key": "updated_at", "label": "updated_at", "type": "unknown"},
    ]
    assert block["rows"] == [
        {"customer": "A", "qty": 10, "price": 12.5, "active": True, "updated_at": None},
        {"customer": "B", "qty": 20, "price": 14.0, "active": False, "updated_at": None},
    ]
    assert block["meta"] == {"row_count": 2, "rendered_row_count": 2, "truncated": False}
    facts = block["details"]["facts"]
    assert facts["source_id"] == "mssql_default"
    assert facts["dialect"] == "tsql"
    assert facts["validated_id"] == "vid-1"
    assert facts["tables"] == ["dbo.offers"]
    assert "customer" in facts["raw_sql"]
    assert "qty" in facts["raw_sql"]
    assert "FROM offers" in facts["raw_sql"]
    tool_message = result.update["messages"][0]
    assert "customer" in tool_message.content
    assert "rendered_row_count" in tool_message.content


def test_execute_validated_sql_stores_ui_tables_in_app_owned_state(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    class Tool:
        def run_structured(self, query):
            return [{"customer": "A", "qty": 10}]

    monkeypatch.setattr(ai_tools, "get_run_query_tool_for_source", lambda source_id: Tool())
    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT customer, qty FROM offers",
                    "source_id": "mssql_default",
                    "dialect": "tsql",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            },
            "sql_result_table_blocks": [{"id": "existing", "type": "data_table"}],
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, Command)
    assert result.update is not None
    assert "sql_result_blocks" not in result.update
    blocks = result.update["sql_result_table_blocks"]
    assert blocks[0] == {"id": "existing", "type": "data_table"}
    assert blocks[1]["type"] == "data_table"
    assert blocks[1]["rows"] == [{"customer": "A", "qty": 10}]
    tool_message = result.update["messages"][0]
    assert "customer" in tool_message.content


def test_execute_validated_sql_formats_raw_sql_for_answer_details(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    class Tool:
        def run_structured(self, query):
            return [{"Дата": "2026-07-06", "Строк": 4}]

    monkeypatch.setattr(ai_tools, "get_run_query_tool_for_source", lambda source_id: Tool())
    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": (
                        'SELECT event_date AS "Дата", count() AS "Строк" '
                        "FROM analytics.sales_fact "
                        "WHERE event_date >= today() - 4 AND event_date <= today() "
                        "GROUP BY event_date ORDER BY event_date ASC"
                    ),
                    "source_id": "clickhouse_default",
                    "dialect": "clickhouse",
                    "tables": ["analytics.sales_fact"],
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, Command)
    raw_sql = result.update["sql_result_table_blocks"][0]["details"]["facts"]["raw_sql"]
    assert "\n" in raw_sql
    assert "FROM analytics.sales_fact" in raw_sql
    assert "WHERE" in raw_sql


def test_submit_model_response_layout_stores_app_owned_layout_and_short_tool_message() -> None:
    layout = {
        "blocks": [
            {"id": "c1", "type": "commentary", "content": "Вот результат."},
            {"id": "p1", "type": "data_table_placeholder", "title": "Продажи"},
        ]
    }
    assert ai_tools.submit_model_response_layout.return_direct is True

    result = getattr(ai_tools.submit_model_response_layout, "func")(layout, "tool-call-1")

    assert isinstance(result, Command)
    assert result.update is not None
    assert result.update["model_response_layout"] == {
        "blocks": [
            {"id": "c1", "type": "commentary", "format": "markdown", "content": "Вот результат."},
            {"id": "p1", "type": "data_table_placeholder", "title": "Продажи"},
        ]
    }
    tool_message = result.update["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.name == "submit_model_response_layout"
    assert tool_message.content == "Final SQL layout submitted."
    assert "Вот результат" not in tool_message.content


def test_submit_model_response_layout_rejects_malformed_layout() -> None:
    result = getattr(ai_tools.submit_model_response_layout, "func")({"blocks": [{"type": "unknown"}]}, "tool-call-1")

    assert result["type"] == "reject"
    assert result["error_code"] == "INVALID_MODEL_RESPONSE_LAYOUT"


def test_submit_model_response_layout_schema_rejects_model_friendly_aliases() -> None:
    schema = ai_tools.SubmitModelResponseLayoutArgs

    try:
        schema.model_validate(
            {
                "layout": {
                    "blocks": [
                        {"type": "commentary", "text": "Посмотрел продажи."},
                        {"type": "data_table_placeholder", "title": "Продажи", "data_table_id": "sql-result-1"},
                    ]
                }
            }
        )
        assert False, "Expected strict final-answer tool schema to reject aliases"
    except Exception as exc:
        error_text = str(exc)

    assert "content" in error_text
    assert "text" in error_text
    assert "data_table_id" in error_text


def test_execute_validated_sql_truncates_structured_rows(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    class Tool:
        def run_structured(self, query):
            return [{"row_number": index} for index in range(55)]

    monkeypatch.setattr(ai_tools, "get_run_query_tool_for_source", lambda source_id: Tool())
    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT * FROM offers",
                    "source_id": "clickhouse_default",
                    "dialect": "clickhouse",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, Command)
    block = result.update["sql_result_table_blocks"][0]
    assert len(block["rows"]) == 50
    assert block["rows"][-1] == {"row_number": 49}
    assert block["meta"] == {"row_count": 55, "rendered_row_count": 50, "truncated": True}


def test_execute_validated_sql_uses_explicit_query_limit_for_rendered_rows(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    class Tool:
        def run_structured(self, query):
            return [{"row_number": index} for index in range(55)]

    monkeypatch.setattr(ai_tools, "get_run_query_tool_for_source", lambda source_id: Tool())
    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT * FROM offers LIMIT 55",
                    "source_id": "clickhouse_default",
                    "dialect": "clickhouse",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, Command)
    block = result.update["sql_result_table_blocks"][0]
    assert len(block["rows"]) == 55
    assert block["rows"][-1] == {"row_number": 54}
    assert block["meta"] == {"row_count": 55, "rendered_row_count": 55, "truncated": False}


def test_execute_validated_sql_falls_back_to_text_when_structured_result_is_string(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    class Tool:
        def run_structured(self, query):
            return "Error: database timeout"

    monkeypatch.setattr(ai_tools, "get_run_query_tool_for_source", lambda source_id: Tool())
    runtime = runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT 1",
                    "source_id": "mssql_default",
                    "dialect": "tsql",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert isinstance(result, Command)
    assert "sql_result_table_blocks" not in result.update
    assert result.update["messages"][0].content == "Error: database timeout"


def test_validate_sql_token_includes_source_and_dialect(monkeypatch) -> None:
    import ai.ai_utils.validate_sql as validate_sql_helpers

    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        validate_sql_helpers, "check_query_is_read_only", lambda query: None
    )
    monkeypatch.setattr(
        ai_tools,
        "load_allowed_source_candidates",
        lambda: [("mssql_default", SQLDialect.MSSQL)],
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
        ai_tools,
        "uuid4",
        lambda: UUID("00000000-0000-0000-0000-000000000125"),
    )
    monkeypatch.setattr(ai_tools.time, "time", lambda: 3000.0)

    runtime = runtime_with_state({"validated_queries": {}})
    result = getattr(ai_tools.validate_sql, "func")(
        "SELECT * FROM sales_orders", runtime
    )

    assert isinstance(result, Command)
    assert result.update is not None
    record = result.update["validated_queries"]["00000000-0000-0000-0000-000000000125"]
    assert record["source_id"] == "mssql_default"
    assert record["dialect"] == "tsql"
    assert record["tables"] == ["dbo.sales_orders"]
