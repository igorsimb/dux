from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import Command

from ai import ai_prompts, ai_tools
from ai.ai_tools_extra import switch_color_theme
from ai.ai_utils import chat_runtime
from ai.ai_utils import streaming
from ai.ai_utils.agent_state import ChatAgentState
from ai.ai_utils.chat_agent import build_chat_agent
from ai.ai_utils.checkpointer import (
    delete_thread_checkpoints,
    get_checkpointer,
    reset_checkpointer,
)
from ai.ai_utils.intent_middleware import build_intent_routing_middleware
from ai.ai_utils.memory_middleware import trim_chat_history_middleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StructuredOutputReadyFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _config(thread_id: str) -> RunnableConfig:
    return cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})


def _validated_input(message: str, thread_id: str) -> ChatAgentState:
    return cast(
        ChatAgentState,
        {
            "messages": [{"role": "user", "content": message}],
            "validated_queries": {
                "token-1": {
                    "sql": "SELECT 1",
                    "source_id": "demo",
                    "dialect": "sqlite",
                    "created_at": 1.0,
                    "expires_at": 2.0,
                    "thread_id": thread_id,
                    "status": "validated",
                }
            },
        },
    )


def _human_and_ai_message_contents(messages: list[Any]) -> list[str]:
    return [message.content for message in messages if isinstance(message, HumanMessage | AIMessage)]


def _request(message: str, tools: list[Any]) -> ModelRequest:
    return ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[HumanMessage(content=message)],
        system_prompt=ai_prompts.SYSTEM_PROMPT_SARCASTIC,
        tools=tools,
        state=cast(ChatAgentState, {"messages": [HumanMessage(content=message)]}),
        runtime=Runtime(),
    )


def test_get_checkpointer_reuses_shared_instance() -> None:
    first = get_checkpointer()
    second = get_checkpointer()

    assert first is second


def test_checkpointer_reset_drops_prior_thread_state_like_process_restart(
    monkeypatch,
) -> None:
    first_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="before reset")]),
        [],
    )
    config = _config("phase2-reset-thread")

    first_agent.invoke(
        cast(Any, _validated_input("hello before reset", "phase2-reset-thread")),
        config=config,
    )

    reset_checkpointer()
    second_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="after reset")]),
        [],
    )
    result = second_agent.invoke(
        {"messages": [{"role": "user", "content": "hello after reset"}]},
        config=config,
    )

    assert second_agent.get_state(config).values["messages"]
    assert _human_and_ai_message_contents(result["messages"]) == [
        "hello after reset",
        "after reset",
    ]
    assert result.get("validated_queries") in ({}, None)


def test_delete_thread_checkpoints_removes_only_requested_thread_state() -> None:
    first_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="thread a reply")]),
        [],
    )
    second_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="thread b reply")]),
        [],
    )
    first_config = _config("phase2-delete-thread-a")
    second_config = _config("phase2-delete-thread-b")

    first_agent.invoke(
        cast(Any, _validated_input("hello a", "phase2-delete-thread-a")),
        config=first_config,
    )
    second_agent.invoke(
        cast(Any, _validated_input("hello b", "phase2-delete-thread-b")),
        config=second_config,
    )

    delete_thread_checkpoints("phase2-delete-thread-a")

    assert first_agent.get_state(first_config).values == {}
    assert (
        second_agent.get_state(second_config).values["validated_queries"]["token-1"][
            "thread_id"
        ]
        == "phase2-delete-thread-b"
    )


def test_fresh_process_does_not_inherit_in_memory_thread_state() -> None:
    agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="parent reply")]),
        [],
    )
    config = _config("phase2-subprocess-thread")
    agent.invoke(
        cast(Any, _validated_input("hello from parent", "phase2-subprocess-thread")),
        config=config,
    )

    command = (
        "import json\n"
        "from ai.ai_utils.chat_agent import build_chat_agent\n"
        "from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel\n"
        "from langchain_core.messages import AIMessage\n"
        "class StructuredOutputReadyFakeModel(FakeMessagesListChatModel):\n"
        "    def bind_tools(self, tools, *, tool_choice=None, **kwargs):\n"
        "        return self\n"
        "agent = build_chat_agent(StructuredOutputReadyFakeModel(responses=[AIMessage(content='child reply')]), [])\n"
        "state = agent.get_state({'configurable': {'thread_id': 'phase2-subprocess-thread'}})\n"
        "print(json.dumps(state.values))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
    )

    assert json.loads(completed.stdout) == {}


def test_build_chat_agent_reuses_shared_checkpointer(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_create_agent(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr("ai.ai_utils.chat_agent.create_agent", fake_create_agent)

    build_chat_agent("model-1", ["tool-1"])
    build_chat_agent("model-2", ["tool-2"])

    assert len(captured) == 2
    assert captured[0]["checkpointer"] is captured[1]["checkpointer"]
    assert len(captured[0]["middleware"]) == 3
    assert captured[0]["middleware"][0] == trim_chat_history_middleware
    assert isinstance(captured[0]["middleware"][1], HumanInTheLoopMiddleware)
    assert captured[0]["middleware"][1].interrupt_on == {
        "ask_user": {"allowed_decisions": ["respond"]}
    }
    assert len(captured[1]["middleware"]) == 3
    assert captured[1]["middleware"][0] == trim_chat_history_middleware
    assert isinstance(captured[1]["middleware"][1], HumanInTheLoopMiddleware)
    assert captured[0]["state_schema"] is ChatAgentState
    assert captured[1]["state_schema"] is ChatAgentState
    assert "response_format" not in captured[0]
    assert "response_format" not in captured[1]


def test_build_chat_agent_uses_app_owned_final_answer_tool_instead_of_response_format(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_create_agent(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr("ai.ai_utils.chat_agent.create_agent", fake_create_agent)

    build_chat_agent("model-1", ["tool-1"])

    assert "response_format" not in captured[0]
    assert captured[0]["tools"] == ["tool-1"]


def test_intent_routing_middleware_uses_no_tools_for_smalltalk_meta() -> None:
    lookup_tool = type("Tool", (), {"name": "lookup_prices"})()
    middleware = build_intent_routing_middleware(
        [lookup_tool, switch_color_theme], ai_prompts.SYSTEM_PROMPT_SARCASTIC
    )
    captured: dict[str, object] = {}

    def handler(request: ModelRequest) -> str:
        captured["tools"] = request.tools
        captured["prompt"] = request.system_prompt
        return "ok"

    result = middleware.wrap_model_call(
        _request("что ты умеешь?", [lookup_tool, switch_color_theme]), handler
    )

    assert result == "ok"
    assert captured == {"tools": [], "prompt": ai_prompts.SYSTEM_PROMPT_SMALLTALK_META}


def test_intent_routing_middleware_exposes_only_theme_tool_for_theme_change() -> None:
    lookup_tool = type("Tool", (), {"name": "lookup_prices"})()
    middleware = build_intent_routing_middleware(
        [lookup_tool, switch_color_theme], ai_prompts.SYSTEM_PROMPT_SARCASTIC
    )
    captured: dict[str, object] = {}

    def handler(request: ModelRequest) -> str:
        captured["tools"] = request.tools
        captured["prompt"] = request.system_prompt
        return "ok"

    result = middleware.wrap_model_call(
        _request("смени тему на nord", [lookup_tool, switch_color_theme]), handler
    )

    assert result == "ok"
    assert captured == {
        "tools": [switch_color_theme],
        "prompt": ai_prompts.SYSTEM_PROMPT_THEME_CHANGE,
    }


def test_intent_routing_middleware_hides_theme_tool_from_sql_agent() -> None:
    lookup_tool = type("Tool", (), {"name": "lookup_prices"})()
    available_tools = [lookup_tool, switch_color_theme]
    middleware = build_intent_routing_middleware(
        available_tools, ai_prompts.SYSTEM_PROMPT_SARCASTIC
    )
    captured: dict[str, object] = {}

    def handler(request: ModelRequest) -> str:
        captured["tools"] = request.tools
        captured["prompt"] = request.system_prompt
        return "ok"

    result = middleware.wrap_model_call(
        _request("привет, покажи топ 10 клиентов по выручке", available_tools), handler
    )

    assert result == "ok"
    assert captured == {
        "tools": [lookup_tool],
        "prompt": ai_prompts.SYSTEM_PROMPT_SARCASTIC,
    }


def test_intent_routing_middleware_applies_overrides_in_async_path() -> None:
    lookup_tool = type("Tool", (), {"name": "lookup_prices"})()
    middleware = build_intent_routing_middleware(
        [lookup_tool, switch_color_theme], ai_prompts.SYSTEM_PROMPT_SARCASTIC
    )

    async def exercise(message: str) -> dict[str, object]:
        captured: dict[str, object] = {}

        async def handler(request: ModelRequest) -> str:
            captured["tools"] = request.tools
            captured["prompt"] = request.system_prompt
            return "ok"

        result = await middleware.awrap_model_call(
            _request(message, [lookup_tool, switch_color_theme]), handler
        )

        assert result == "ok"
        return captured

    assert asyncio.run(exercise("what can you do?")) == {
        "tools": [],
        "prompt": ai_prompts.SYSTEM_PROMPT_SMALLTALK_META,
    }
    assert asyncio.run(exercise("switch the theme to nord")) == {
        "tools": [switch_color_theme],
        "prompt": ai_prompts.SYSTEM_PROMPT_THEME_CHANGE,
    }


def test_intent_routing_middleware_logs_selected_intent_mode_without_raw_user_text(
    monkeypatch,
) -> None:
    lookup_tool = type("Tool", (), {"name": "lookup_prices"})()
    middleware = build_intent_routing_middleware(
        [lookup_tool, switch_color_theme], ai_prompts.SYSTEM_PROMPT_SARCASTIC
    )
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        "ai.ai_utils.intent_middleware.logger",
        type(
            "Logger",
            (),
            {
                "debug": lambda self, message, *args: debug_calls.append(
                    (str(message), args)
                )
            },
        )(),
    )

    def handler(request: ModelRequest) -> str:
        return "ok"

    result = middleware.wrap_model_call(
        _request("what can you do?", [lookup_tool, switch_color_theme]), handler
    )

    assert result == "ok"
    assert len(debug_calls) == 1
    message, args = debug_calls[0]
    assert args == ()
    assert "chat.intent" in message
    assert "selected" in message
    assert "smalltalk_meta" in message
    assert "what can you do" not in message


def test_build_chat_agent_preserves_validated_queries_in_custom_state() -> None:
    agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="first reply")]),
        [],
    )
    config = _config("phase2-state-thread")

    result = agent.invoke(
        cast(Any, _validated_input("hello", "phase2-state-thread")),
        config=config,
    )

    assert result["validated_queries"]["token-1"]["sql"] == "SELECT 1"
    assert (
        agent.get_state(config).values["validated_queries"]["token-1"]["status"]
        == "validated"
    )


def test_same_thread_id_reuses_checkpointed_state_across_agent_invocations() -> None:
    first_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="first reply")]),
        [],
    )
    second_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="second reply")]),
        [],
    )
    config = _config("phase2-shared-thread")

    first_agent.invoke(
        cast(Any, _validated_input("hello", "phase2-shared-thread")),
        config=config,
    )
    result = second_agent.invoke(
        {"messages": [{"role": "user", "content": "again"}]},
        config=config,
    )

    assert result["validated_queries"]["token-1"]["thread_id"] == "phase2-shared-thread"
    assert _human_and_ai_message_contents(result["messages"]) == [
        "hello",
        "first reply",
        "again",
        "second reply",
    ]


def test_different_thread_ids_stay_isolated_across_agent_invocations() -> None:
    first_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="first reply")]),
        [],
    )
    second_agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="second reply")]),
        [],
    )
    first_config = _config("phase2-isolated-thread-a")
    second_config = _config("phase2-isolated-thread-b")

    first_agent.invoke(
        cast(Any, _validated_input("hello", "phase2-isolated-thread-a")),
        config=first_config,
    )
    result = second_agent.invoke(
        {"messages": [{"role": "user", "content": "other thread"}]},
        config=second_config,
    )

    assert result.get("validated_queries") in ({}, None)
    assert _human_and_ai_message_contents(result["messages"]) == [
        "other thread",
        "second reply",
    ]


def test_streaming_consumers_work_with_agent_builder(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    agent = build_chat_agent(
        StructuredOutputReadyFakeModel(responses=[AIMessage(content="streamed reply")]),
        [],
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            "hello",
            "robot-phase2-stream",
            "phase2-stream-thread",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {"kind": "token", "text": "streamed reply"} in items


def test_chat_agent_preserves_sql_result_blocks_for_structured_placeholder_resolution(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")
    reset_checkpointer()

    @tool
    def capture_sql_result(tool_runtime: ToolRuntime) -> Command:
        """Capture a deterministic structured SQL result block."""

        return Command(
            update={
                "sql_result_table_blocks": [
                    {
                        "id": "sql-result-1",
                        "type": "data_table",
                        "columns": [{"key": "order_id", "label": "order_id", "type": "string"}],
                        "rows": [{"order_id": "ORD-123"}],
                        "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
                    }
                ],
                "messages": [
                    ToolMessage(
                        content="captured",
                        tool_call_id=tool_runtime.tool_call_id,
                        name="capture_sql_result",
                    )
                ],
            }
        )

    agent = build_chat_agent(
        StructuredOutputReadyFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call-1", "name": "capture_sql_result", "args": {}}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-layout-1",
                            "name": "submit_model_response_layout",
                            "args": {
                                "layout": {
                                    "blocks": [
                                        {"id": "c1", "type": "commentary", "content": "Here is the table."},
                                        {"id": "p1", "type": "data_table_placeholder", "title": "Orders"},
                                    ]
                                }
                            },
                        }
                    ],
                ),
            ]
        ),
        [capture_sql_result, ai_tools.submit_model_response_layout],
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            "show orders",
            "robot-sql-result-blocks",
            "thread-sql-result-blocks",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    block_event = next(item for item in items if isinstance(item, dict) and item.get("kind") == "blocks")
    assert block_event["blocks"][1]["type"] == "data_table"
    assert block_event["blocks"][1]["title"] == "Orders"
    assert block_event["blocks"][1]["rows"] == [{"order_id": "ORD-123"}]


def test_run_chat_session_uses_chat_agent_builder(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        chat_runtime,
        "build_model_and_tools",
        lambda stream_timeout_seconds: ("model", ["tool"], "gpt-test"),
    )

    def fake_build_chat_agent(model, tools):
        captured["builder_args"] = (model, tools)
        return "new-agent"

    monkeypatch.setattr(chat_runtime, "build_chat_agent", fake_build_chat_agent)

    async def fake_stream(agent, user_text, robot_id, thread_id, queue, *, model_name) -> None:
        captured["stream_agent"] = agent
        captured["thread_id"] = thread_id

    monkeypatch.setattr(chat_runtime, "produce_agent_stream_async", fake_stream)

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        await chat_runtime.run_chat_session(
            user_text="test",
            robot_id="robot-async-2",
            thread_id="thread-async-2",
            queue=queue,
            stream_timeout_seconds=10,
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert captured == {
        "builder_args": ("model", ["tool"]),
        "stream_agent": "new-agent",
        "thread_id": "thread-async-2",
    }
    assert items == [None]
