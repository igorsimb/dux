from __future__ import annotations

import pytest

from ai.ai_utils import chat_runtime
from ai.sql_dialect import SQLDialect


def test_build_model_and_tools_registers_query_tools_for_all_sources(
    monkeypatch,
) -> None:
    init_chat_model_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        chat_runtime,
        "load_allowed_source_candidates",
        lambda: [
            ("clickhouse_default", SQLDialect.CLICKHOUSE),
            ("mssql_default", SQLDialect.MSSQL),
        ],
    )
    monkeypatch.setattr(chat_runtime, "get_model_name", lambda: "gpt-test")
    monkeypatch.setattr(chat_runtime, "get_openai_proxy", lambda: None)
    monkeypatch.setattr(chat_runtime, "get_enable_streaming", lambda: True)
    monkeypatch.setattr(
        chat_runtime,
        "init_chat_model",
        lambda **kwargs: init_chat_model_calls.append(kwargs) or "model",
    )

    db_calls: list[str] = []
    monkeypatch.setattr(
        chat_runtime,
        "get_sql_database_for_source",
        lambda source_id: db_calls.append(source_id) or f"db:{source_id}",
    )

    registrations: list[tuple[str, str]] = []
    factory_registrations: list[str] = []
    monkeypatch.setattr(chat_runtime, "clear_run_query_tools", lambda: None)
    monkeypatch.setattr(
        chat_runtime,
        "set_run_query_tool_for_source",
        lambda source_id, tool: registrations.append((source_id, tool)),
    )
    monkeypatch.setattr(
        chat_runtime,
        "set_run_query_tool_factory_for_source",
        lambda source_id, factory: factory_registrations.append(source_id),
    )
    monkeypatch.setattr(
        chat_runtime,
        "build_guarded_sql_tools",
        lambda db, model: (["sql_tools"], f"list:{db}", f"schema:{db}", f"run:{db}"),
    )

    model, tools, enable_streaming, model_name = chat_runtime.build_model_and_tools(120)

    assert model == "model"
    assert model_name == "gpt-test"
    assert enable_streaming is True
    assert init_chat_model_calls == [
        {
            "model": "gpt-test",
            "use_responses_api": True,
            "openai_proxy": None,
            "streaming": True,
            "stream_usage": True,
            "request_timeout": 120,
            "max_retries": 1,
        }
    ]
    assert db_calls == []
    assert registrations == []
    assert factory_registrations == ["clickhouse_default", "mssql_default"]
    assert "list:db:clickhouse_default" not in tools
    assert "schema:db:clickhouse_default" not in tools
    assert chat_runtime.ask_user in tools
    assert chat_runtime.submit_model_response_layout in tools


def test_build_model_and_tools_omits_schema_tools_for_single_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        chat_runtime,
        "load_allowed_source_candidates",
        lambda: [("clickhouse_default", SQLDialect.CLICKHOUSE)],
    )
    monkeypatch.setattr(chat_runtime, "get_model_name", lambda: "gpt-test")
    monkeypatch.setattr(chat_runtime, "get_openai_proxy", lambda: None)
    monkeypatch.setattr(chat_runtime, "get_enable_streaming", lambda: True)
    monkeypatch.setattr(chat_runtime, "init_chat_model", lambda **kwargs: "model")
    monkeypatch.setattr(
        chat_runtime,
        "get_sql_database_for_source",
        lambda source_id: f"db:{source_id}",
    )
    monkeypatch.setattr(chat_runtime, "clear_run_query_tools", lambda: None)
    monkeypatch.setattr(
        chat_runtime, "set_run_query_tool_for_source", lambda source_id, tool: None
    )
    monkeypatch.setattr(
        chat_runtime,
        "set_run_query_tool_factory_for_source",
        lambda source_id, factory: None,
    )
    monkeypatch.setattr(
        chat_runtime,
        "build_guarded_sql_tools",
        lambda db, model: (["sql_tools"], f"list:{db}", f"schema:{db}", f"run:{db}"),
    )

    _model, tools, _enable_streaming, _model_name = chat_runtime.build_model_and_tools(
        120
    )

    assert "list:db:clickhouse_default" not in tools
    assert "schema:db:clickhouse_default" not in tools
    assert chat_runtime.ask_user in tools
    assert chat_runtime.submit_model_response_layout in tools


def test_build_model_and_tools_raises_when_no_source_candidates(monkeypatch) -> None:
    monkeypatch.setattr(chat_runtime, "load_allowed_source_candidates", lambda: [])

    with pytest.raises(ValueError) as exc_info:
        chat_runtime.build_model_and_tools(120)

    assert "No allowlisted source+dialect candidates configured" in str(exc_info.value)
