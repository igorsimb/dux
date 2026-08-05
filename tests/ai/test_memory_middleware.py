from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from ai.ai_utils import memory_middleware
from ai.ai_utils.agent_state import ChatAgentState


def _state(messages: list[Any]) -> ChatAgentState:
    return cast(
        ChatAgentState,
        {
            "messages": messages,
            "validated_queries": {
                "token-1": {
                    "sql": "SELECT 1",
                    "source_id": "demo",
                    "dialect": "sqlite",
                    "created_at": 1.0,
                    "expires_at": 2.0,
                    "thread_id": "phase3-thread",
                    "status": "validated",
                }
            },
        },
    )


def _content(message: Any) -> str:
    return str(getattr(message, "content", ""))


def test_trim_chat_history_returns_none_for_empty_history() -> None:
    state = _state([])

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is None


def test_trim_chat_history_does_not_rewrite_small_single_turn_history(
    monkeypatch,
) -> None:
    monkeypatch.setattr(memory_middleware, "MAX_HISTORY_TOKENS", 100)

    state = _state([HumanMessage(content="hi")])

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is None


def test_trim_chat_history_removes_old_messages_and_preserves_recent_messages(
    monkeypatch,
) -> None:
    monkeypatch.setattr(memory_middleware, "MAX_HISTORY_TOKENS", 12)

    state = _state(
        [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
        ]
    )

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is not None
    assert isinstance(result["messages"][0], RemoveMessage)
    assert result["messages"][0].id == REMOVE_ALL_MESSAGES
    assert [_content(message) for message in result["messages"][1:]] == [
        "recent question"
    ]


def test_trim_chat_history_respects_message_boundaries_for_tool_turns(
    monkeypatch,
) -> None:
    monkeypatch.setattr(memory_middleware, "MAX_HISTORY_TOKENS", 50)

    state = _state(
        [
            HumanMessage(content="old question"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call-old", "name": "lookup", "args": {"q": "old"}}],
            ),
            ToolMessage(content="old result", tool_call_id="call-old"),
            HumanMessage(content="recent question"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call-new", "name": "lookup", "args": {"q": "new"}}],
            ),
            ToolMessage(content="recent result", tool_call_id="call-new"),
        ]
    )

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is not None
    trimmed_messages = result["messages"][1:]
    assert [type(message).__name__ for message in trimmed_messages] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert [_content(message) for message in trimmed_messages] == [
        "recent question",
        "",
        "recent result",
    ]
    assert trimmed_messages[1].tool_calls[0]["id"] == "call-new"
    assert trimmed_messages[2].tool_call_id == "call-new"


def test_trim_chat_history_keeps_validated_queries_state_untouched(monkeypatch) -> None:
    monkeypatch.setattr(memory_middleware, "MAX_HISTORY_TOKENS", 12)

    state = _state(
        [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
        ]
    )
    original_validated_queries = state.get("validated_queries")
    assert original_validated_queries is not None

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is not None
    assert list(result) == ["messages"]
    current_validated_queries = state.get("validated_queries")
    assert current_validated_queries is original_validated_queries
    assert current_validated_queries is not None
    assert current_validated_queries["token-1"]["sql"] == "SELECT 1"


def test_trim_chat_history_keeps_oversized_final_human_message(monkeypatch) -> None:
    monkeypatch.setattr(memory_middleware, "MAX_HISTORY_TOKENS", 1)

    state = _state([HumanMessage(content="x" * 500)])

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is None


def test_trim_chat_history_keeps_oversized_final_tool_turn(monkeypatch) -> None:
    monkeypatch.setattr(memory_middleware, "MAX_HISTORY_TOKENS", 1)

    state = _state(
        [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call-new", "name": "lookup", "args": {"q": "new"}}],
            ),
            ToolMessage(content="x" * 500, tool_call_id="call-new"),
        ]
    )

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is not None
    trimmed_messages = result["messages"][1:]
    assert [type(message).__name__ for message in trimmed_messages] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert trimmed_messages[0].content == "recent question"
    assert trimmed_messages[1].tool_calls[0]["id"] == "call-new"
    assert trimmed_messages[2].tool_call_id == "call-new"


def test_trim_chat_history_logs_trim_decision_without_raw_message_content(
    monkeypatch,
) -> None:
    monkeypatch.setattr(memory_middleware, "MAX_HISTORY_TOKENS", 12)
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        memory_middleware.logger,
        "debug",
        lambda message, *args: debug_calls.append((str(message), args)),
    )

    state = _state(
        [
            HumanMessage(content="secret old question"),
            AIMessage(content="secret old answer"),
            HumanMessage(content="recent question"),
        ]
    )

    result = memory_middleware.trim_chat_history(state, Runtime())

    assert result is not None
    assert len(debug_calls) == 1
    message, args = debug_calls[0]
    assert args == ()
    assert "chat.memory" in message
    assert "trim_applied" in message
    assert "before=3" in message
    assert "after=1" in message
    assert "tokens_before=" in message
    assert "tokens_after=" in message
    assert f"counter={memory_middleware.TOKEN_COUNTER_NAME}" in message
    assert "secret old question" not in message
    assert "secret old answer" not in message
