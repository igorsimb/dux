"""ai_tools progress emission tests (stage-flow focused).

Flow:
- Stub writer context (`get_stream_writer`) and capture emissions via `show_progress_message`.
- Stub message selectors to deterministic `group:stage` markers.
- Drive specific tool branches and assert emitted stage order (start/retry/ok, etc.).
"""

from types import SimpleNamespace

from ai import ai_tools
from ai.ai_utils import validate_sql as validate_sql_helpers
from ai.sql_dialect import SQLDialect


def _runtime_with_state(state: dict | None = None) -> SimpleNamespace:
    """Build minimal ToolRuntime-like object used by direct tool function calls."""
    return SimpleNamespace(
        config={"configurable": {"thread_id": "t1"}},
        state=state or {},
        tool_call_id="tool-call-1",
    )


def test_get_table_metadata_emits_start_and_not_found(monkeypatch) -> None:
    """Metadata tool emits START first, then NOT_FOUND on missing table."""
    emitted: list[str] = []
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "show_progress_message",
        lambda *, writer, stage: emitted.append(stage),
    )
    monkeypatch.setattr(
        ai_tools,
        "get_table_metadata_message",
        lambda stage: f"metadata:{stage.value}",
    )
    monkeypatch.setattr(ai_tools, "load_table_metadata", lambda: {})

    result = getattr(ai_tools.get_table_metadata, "func")("missing.table")

    assert result == "Table not found or not allowed."
    assert emitted == ["metadata:start", "metadata:not_found"]


def test_validate_sql_readonly_reject_emits_start_then_retry(monkeypatch) -> None:
    """Validation emits START and RETRY when readonly guard rejects query."""
    emitted: list[str] = []
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "show_progress_message",
        lambda *, writer, stage: emitted.append(stage),
    )
    monkeypatch.setattr(
        ai_tools,
        "validate_sql_message",
        lambda stage: f"validate:{stage.value}",
    )
    monkeypatch.setattr(
        validate_sql_helpers,
        "check_query_is_read_only",
        lambda query: {
            "type": "reject",
            "error_code": "READ_ONLY_GUARD",
            "reason": "bad",
        },
    )

    result = getattr(ai_tools.validate_sql, "func")("SELECT 1", _runtime_with_state())

    assert isinstance(result, dict)
    assert result.get("error_code") == "READ_ONLY_GUARD"
    assert emitted == ["validate:start", "validate:retry"]


def test_validate_sql_success_emits_start_then_ok(monkeypatch) -> None:
    """Validation emits START then OK when all checks pass."""
    emitted: list[str] = []
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "show_progress_message",
        lambda *, writer, stage: emitted.append(stage),
    )
    monkeypatch.setattr(
        ai_tools,
        "validate_sql_message",
        lambda stage: f"validate:{stage.value}",
    )
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
            "SELECT 1",
            ["analytics.customers"],
            None,
        ),
    )

    result = getattr(ai_tools.validate_sql, "func")("SELECT 1", _runtime_with_state())

    assert not isinstance(result, dict)
    assert emitted == ["validate:start", "validate:ok"]


def test_execute_validated_sql_unknown_id_emits_start_then_problem(monkeypatch) -> None:
    """Execution emits START then PROBLEM when validated_id is unknown."""
    emitted: list[str] = []
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "show_progress_message",
        lambda *, writer, stage: emitted.append(stage),
    )
    monkeypatch.setattr(
        ai_tools,
        "execute_validated_sql_message",
        lambda stage: f"execute:{stage.value}",
    )

    result = getattr(ai_tools.execute_validated_sql, "func")(
        "missing", _runtime_with_state()
    )

    assert isinstance(result, dict)
    assert result.get("error_code") == "UNKNOWN_VALIDATED_ID"
    assert emitted == ["execute:start", "execute:problem"]


def test_execute_validated_sql_without_query_tool_emits_connecting(monkeypatch) -> None:
    """Execution emits DB_CONNECTING when SQL query tool is unavailable."""
    emitted: list[str] = []
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "show_progress_message",
        lambda *, writer, stage: emitted.append(stage),
    )
    monkeypatch.setattr(
        ai_tools,
        "execute_validated_sql_message",
        lambda stage: f"execute:{stage.value}",
    )
    monkeypatch.setattr(
        ai_tools, "get_run_query_tool_for_source", lambda source_id: None
    )
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)
    runtime = _runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT 1",
                    "source_id": "clickhouse_default",
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
    assert emitted == ["execute:start", "execute:db_connecting"]


def test_execute_validated_sql_success_emits_waiting_then_final_analysis(
    monkeypatch,
) -> None:
    """Successful execution emits DB_WAITING before FINAL_ANALYSIS."""
    emitted: list[str] = []
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "show_progress_message",
        lambda *, writer, stage: emitted.append(stage),
    )
    monkeypatch.setattr(
        ai_tools,
        "execute_validated_sql_message",
        lambda stage: f"execute:{stage.value}",
    )
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)

    class _Tool:
        def run_structured(self, query):
            return [{"answer": 1}]

    monkeypatch.setattr(
        ai_tools, "get_run_query_tool_for_source", lambda source_id: _Tool()
    )
    runtime = _runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT 1",
                    "source_id": "clickhouse_default",
                    "created_at": 0.0,
                    "expires_at": 200.0,
                    "thread_id": "t1",
                    "status": "validated",
                }
            }
        }
    )

    result = getattr(ai_tools.execute_validated_sql, "func")("vid-1", runtime)

    assert not isinstance(result, dict)
    assert emitted == [
        "execute:start",
        "execute:db_waiting",
        "execute:final_analysis",
    ]


def test_execute_validated_sql_missing_source_id_rejects(monkeypatch) -> None:
    emitted: list[str] = []
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "show_progress_message",
        lambda *, writer, stage: emitted.append(stage),
    )
    monkeypatch.setattr(
        ai_tools,
        "execute_validated_sql_message",
        lambda stage: f"execute:{stage.value}",
    )
    monkeypatch.setattr(ai_tools.time, "time", lambda: 100.0)
    runtime = _runtime_with_state(
        {
            "validated_queries": {
                "vid-1": {
                    "sql": "SELECT 1",
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
    assert result.get("error_code") == "VALIDATED_QUERY_SOURCE_MISSING"
    assert emitted == ["execute:start", "execute:problem"]
