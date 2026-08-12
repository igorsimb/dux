from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.types import Command, Interrupt

from ai import ai_tools
from ai.ai_utils import streaming
from ai.ai_utils.chat_agent import build_chat_agent
from ai.ai_utils.checkpointer import reset_checkpointer
from ai.ai_utils.memory_middleware import trim_chat_history_middleware
from ai.ai_utils.structured_output_blocks import AgentCommentaryResponse


def test_build_usage_event_aggregates_usage_across_models() -> None:
    usage_event = streaming.build_usage_event(
        {
            "gpt-5.4": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": {"cache_read": 2},
            },
            "gpt-5.4-mini": {
                "input_tokens": 4,
                "output_tokens": 6,
                "total_tokens": 10,
                "output_token_details": {"reasoning": 1},
            },
        },
        "gpt-5.4",
    )

    assert usage_event == {
        "kind": "usage",
        "model": "gpt-5.4",
        "pricing_model": "gpt-5.4",
        "cost_usd": 0.0001955,
        "usage_metadata": {
            "input_tokens": 14,
            "output_tokens": 11,
            "total_tokens": 25,
            "input_token_details": {"cache_read": 2},
            "output_token_details": {"reasoning": 1},
        },
    }


def test_build_usage_event_prefers_single_observed_model_name() -> None:
    usage_event = streaming.build_usage_event(
        {
            "gpt-5.4-2026-03-01": {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            }
        },
        "gpt-5.4",
    )

    assert usage_event == {
        "kind": "usage",
        "model": "gpt-5.4-2026-03-01",
        "pricing_model": "gpt-5.4",
        "cost_usd": 3.75e-05,
        "usage_metadata": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


def test_build_usage_event_omits_cost_for_unknown_model() -> None:
    usage_event = streaming.build_usage_event(
        {"unknown-model": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        "unknown-model",
    )

    assert usage_event == {
        "kind": "usage",
        "model": "unknown-model",
        "pricing_model": "unknown-model",
        "usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def test_extract_model_response_layout_uses_app_owned_state_only() -> None:
    response = streaming.extract_model_response_layout(
        {
            "model_response_layout": {
                "blocks": [{"id": "c1", "type": "commentary", "content": "Готово."}]
            },
        }
    )

    assert isinstance(response, AgentCommentaryResponse)
    assert response.blocks[0].id == "c1"


def test_build_final_response_blocks_resolves_placeholders_positionally() -> None:
    response = AgentCommentaryResponse.model_validate(
        {
            "blocks": [
                {"id": "c1", "type": "commentary", "content": "Вот результат."},
                {"id": "p1", "type": "data_table_placeholder", "title": "Топ клиентов"},
            ]
        }
    )
    data = {
        "sql_result_table_blocks": [
            {
                "id": "sql-result-1",
                "type": "data_table",
                "columns": [{"key": "customer", "label": "customer", "type": "string"}],
                "rows": [{"customer": "A"}],
                "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
            }
        ]
    }

    blocks = streaming.build_final_response_blocks(data, response)

    assert blocks[0] == {
        "id": "c1",
        "type": "commentary",
        "format": "markdown",
        "content": "Вот результат.",
    }
    assert blocks[1]["type"] == "data_table"
    assert blocks[1]["title"] == "Топ клиентов"
    assert blocks[1]["rows"] == [{"customer": "A"}]


def test_build_final_response_blocks_reads_sql_result_table_blocks() -> None:
    response = AgentCommentaryResponse.model_validate(
        {
            "blocks": [
                {"id": "c1", "type": "commentary", "content": "Вот результат."},
                {"id": "p1", "type": "data_table_placeholder", "title": "Топ клиентов"},
            ]
        }
    )
    data = {
        "sql_result_table_blocks": [
            {
                "id": "sql-result-1",
                "type": "data_table",
                "columns": [{"key": "customer", "label": "customer", "type": "string"}],
                "rows": [{"customer": "A"}],
                "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
            }
        ],
    }

    blocks = streaming.build_final_response_blocks(data, response)

    assert blocks[1]["type"] == "data_table"
    assert blocks[1]["title"] == "Топ клиентов"
    assert blocks[1]["rows"] == [{"customer": "A"}]


def test_build_final_response_blocks_copies_placeholder_notes_to_matched_table() -> None:
    response = AgentCommentaryResponse.model_validate(
        {
            "blocks": [
                {
                    "id": "p1",
                    "type": "data_table_placeholder",
                    "title": "Топ клиентов",
                    "notes": [
                        {"label": "Период", "value": "последние 30 дней"},
                        {"label": "Метрика", "value": "количество продаж"},
                    ],
                }
            ]
        }
    )
    data = {
        "sql_result_table_blocks": [
            {
                "id": "sql-result-1",
                "type": "data_table",
                "columns": [{"key": "customer", "label": "customer", "type": "string"}],
                "rows": [{"customer": "A"}],
                "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
            }
        ]
    }

    blocks = streaming.build_final_response_blocks(data, response)

    assert blocks == [
        {
            "id": "sql-result-1",
            "type": "data_table",
            "title": "Топ клиентов",
            "columns": [{"key": "customer", "label": "customer", "type": "string"}],
            "rows": [{"customer": "A"}],
            "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
            "details": {
                "notes": [
                    {"label": "Период", "value": "последние 30 дней"},
                    {"label": "Метрика", "value": "количество продаж"},
                ]
            },
        }
    ]


def test_build_final_response_blocks_preserves_backend_facts_when_copying_notes() -> None:
    response = AgentCommentaryResponse.model_validate(
        {
            "blocks": [
                {
                    "id": "p1",
                    "type": "data_table_placeholder",
                    "notes": [{"label": "Период", "value": "последние 30 дней"}],
                }
            ]
        }
    )
    data = {
        "sql_result_table_blocks": [
            {
                "id": "sql-result-1",
                "type": "data_table",
                "columns": [],
                "rows": [],
                "meta": {"row_count": 0, "rendered_row_count": 0, "truncated": False},
                "details": {
                    "facts": {
                        "source_id": "mssql_default",
                        "dialect": "tsql",
                        "validated_id": "vid-1",
                        "tables": ["dbo.customer_orders"],
                        "raw_sql": "SELECT * FROM dbo.customer_orders",
                    }
                },
            }
        ]
    }

    blocks = streaming.build_final_response_blocks(data, response)

    details = blocks[0]["details"]
    assert details["facts"]["source_id"] == "mssql_default"
    assert details["facts"]["raw_sql"] == "SELECT * FROM dbo.customer_orders"
    assert details["notes"] == [{"label": "Период", "value": "последние 30 дней"}]


def test_build_final_response_blocks_appends_unused_sql_result_blocks() -> None:
    response = AgentCommentaryResponse.model_validate(
        {"blocks": [{"id": "c1", "type": "commentary", "content": "Готово."}]}
    )
    data = {
        "sql_result_table_blocks": [
            {
                "id": "sql-result-1",
                "type": "data_table",
                "columns": [],
                "rows": [],
                "meta": {"row_count": 0, "rendered_row_count": 0, "truncated": False},
            }
        ]
    }

    blocks = streaming.build_final_response_blocks(data, response)

    assert [block["type"] for block in blocks] == ["commentary", "data_table"]


def test_build_final_response_blocks_resolves_empty_sql_result() -> None:
    response = AgentCommentaryResponse.model_validate(
        {"blocks": [{"id": "p1", "type": "data_table_placeholder", "title": "Нет результатов"}]}
    )
    data = {
        "sql_result_table_blocks": [
            {
                "id": "sql-result-1",
                "type": "data_table",
                "columns": [],
                "rows": [],
                "meta": {"row_count": 0, "rendered_row_count": 0, "truncated": False},
            }
        ]
    }

    blocks = streaming.build_final_response_blocks(data, response)

    assert blocks == [
        {
            "id": "sql-result-1",
            "type": "data_table",
            "title": "Нет результатов",
            "columns": [],
            "rows": [],
            "meta": {"row_count": 0, "rendered_row_count": 0, "truncated": False},
        }
    ]


def test_build_final_response_blocks_skips_missing_table_placeholder(monkeypatch) -> None:
    response = AgentCommentaryResponse.model_validate(
        {"blocks": [{"id": "p1", "type": "data_table_placeholder", "title": "Нет таблицы"}]}
    )
    warning_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        streaming.logger,
        "warning",
        lambda message, *args: warning_calls.append((str(message), args)),
    )

    blocks = streaming.build_final_response_blocks({}, response)

    assert blocks == []
    assert len(warning_calls) == 1
    message, args = warning_calls[0]
    assert args == ()
    assert "placeholder_unmatched" in message
    assert "p1" in message
    assert "available_sql_blocks=0" in message


def test_build_final_response_blocks_skips_malformed_sql_result_block(monkeypatch) -> None:
    response = AgentCommentaryResponse.model_validate(
        {"blocks": [{"id": "c1", "type": "commentary", "content": "Готово."}]}
    )
    warning_calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        streaming.logger,
        "warning",
        lambda message, *args: warning_calls.append((str(message), args)),
    )

    blocks = streaming.build_final_response_blocks(
        {"sql_result_table_blocks": [{"id": "broken", "type": "data_table"}]}, response
    )

    assert blocks == [{"id": "c1", "type": "commentary", "format": "markdown", "content": "Готово."}]
    assert len(warning_calls) == 1
    message, args = warning_calls[0]
    assert args == ()
    assert "sql_block_malformed" in message
    assert "index=0" in message


def test_extract_ask_user_questions_supports_args_and_arguments() -> None:
    interrupts = (
        Interrupt(
            value={
                "action_requests": [
                    {"name": "ask_user", "args": {"question": "Какой период?"}},
                    {"name": "other_tool", "args": {"question": "ignore"}},
                    {"name": "ask_user", "arguments": {"question": "Сколько строк?"}},
                ]
            }
        ),
    )

    assert streaming.extract_ask_user_questions_from_interrupts(interrupts) == [
        "Какой период?",
        "Сколько строк?",
    ]


def test_build_agent_graph_input_resumes_pending_ask_user_interrupt() -> None:
    class FakeAgent:
        async def aget_state(self, config):
            assert config["configurable"] == {"thread_id": "thread-hitl"}
            return SimpleNamespace(
                interrupts=(
                    Interrupt(
                        value={
                            "action_requests": [
                                {"name": "ask_user", "args": {"question": "Какой период?"}}
                            ]
                        }
                    ),
                )
            )

    graph_input = asyncio.run(
        streaming.build_agent_graph_input(
            FakeAgent(), "последние 30 дней", {"configurable": {"thread_id": "thread-hitl"}}
        )
    )

    assert isinstance(graph_input, Command)
    assert graph_input.goto == ()
    assert graph_input.resume == {
        "decisions": [{"type": "respond", "message": "последние 30 дней"}]
    }


def test_build_agent_graph_input_resumes_multiple_pending_ask_user_interrupts() -> None:
    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(
                interrupts=(
                    Interrupt(
                        value={
                            "action_requests": [
                                {"name": "ask_user", "args": {"question": "Какой период?"}},
                                {"name": "ask_user", "args": {"question": "Сколько строк?"}},
                            ]
                        }
                    ),
                )
            )

    graph_input = asyncio.run(
        streaming.build_agent_graph_input(FakeAgent(), "30 дней и 20 строк", {"configurable": {"thread_id": "t"}})
    )

    assert isinstance(graph_input, Command)
    assert graph_input.goto == ()
    assert graph_input.resume == {
        "decisions": [
            {"type": "respond", "message": "30 дней и 20 строк"},
            {"type": "respond", "message": "30 дней и 20 строк"},
        ]
    }
    assert streaming.build_ask_user_message(["Какой период?", "Сколько строк?"]) == "Какой период?\n\nСколько строк?"


def test_build_agent_graph_input_resets_sql_result_blocks_for_fresh_turn() -> None:
    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(interrupts=())

    graph_input = asyncio.run(
        streaming.build_agent_graph_input(FakeAgent(), "hello", {"configurable": {"thread_id": "t"}})
    )

    assert graph_input == {
        "messages": [streaming.HumanMessage(content="hello")],
        "sql_result_table_blocks": [],
        "model_response_layout": None,
    }


def test_build_agent_graph_input_resets_app_owned_render_state_for_fresh_turn() -> None:
    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(
                values={
                    "model_response_layout": {
                        "blocks": [{"id": "old", "type": "commentary", "content": "Старый ответ."}]
                    },
                    "sql_result_table_blocks": [{"id": "old-table", "type": "data_table"}],
                    "validated_queries": {"vid-1": {"sql": "SELECT 1"}},
                },
                interrupts=(),
            )

    graph_input = asyncio.run(
        streaming.build_agent_graph_input(FakeAgent(), "hello", {"configurable": {"thread_id": "t"}})
    )

    assert graph_input == {
        "messages": [streaming.HumanMessage(content="hello")],
        "sql_result_table_blocks": [],
        "model_response_layout": None,
    }


def test_produce_agent_stream_async_renders_fresh_sql_table_after_prior_table(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {}

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    old_layout = {"blocks": [{"id": "old-p1", "type": "data_table_placeholder", "title": "Старая таблица"}]}
    old_table = {
        "id": "sql-result-1",
        "type": "data_table",
        "columns": [{"key": "old", "label": "old", "type": "string"}],
        "rows": [{"old": "A"}],
        "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
    }
    fresh_layout = {"blocks": [{"id": "p1", "type": "data_table_placeholder", "title": "Новая таблица"}]}
    fresh_table = {
        "id": "sql-result-1",
        "type": "data_table",
        "columns": [{"key": "customer", "label": "customer", "type": "string"}],
        "rows": [{"customer": "A"}],
        "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
    }

    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(
                values={
                    "messages": [streaming.AIMessage(content="old")],
                    "model_response_layout": old_layout,
                    "sql_result_table_blocks": [old_table],
                },
                interrupts=(),
            )

        async def astream(self, graph_input, config, stream_mode, **kwargs):
            assert graph_input == {
                "messages": [streaming.HumanMessage(content="new")],
                "sql_result_table_blocks": [],
                "model_response_layout": None,
            }
            yield "values", {"model_response_layout": fresh_layout, "sql_result_table_blocks": [fresh_table]}

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "new",
            "robot-fresh-table",
            "thread-fresh-table",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {"kind": "blocks", "blocks": [{**fresh_table, "title": "Новая таблица"}]} in items


def test_build_agent_graph_input_does_not_force_model_goto_for_app_owned_layout() -> None:
    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(
                values={
                    "model_response_layout": {
                        "blocks": [{"id": "old", "type": "commentary", "content": "Старый ответ."}]
                    },
                    "sql_result_table_blocks": [],
                },
                interrupts=(
                    Interrupt(
                        value={
                            "action_requests": [
                                {"name": "ask_user", "args": {"question": "Какой период?"}}
                            ]
                        }
                    ),
                ),
            )

    graph_input = asyncio.run(
        streaming.build_agent_graph_input(FakeAgent(), "последние 30 дней", {"configurable": {"thread_id": "t"}})
    )

    assert isinstance(graph_input, Command)
    assert graph_input.goto == ()
    assert graph_input.resume == {"decisions": [{"type": "respond", "message": "последние 30 дней"}]}


def test_produce_agent_stream_async_emits_ask_user_interrupt_question(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {}

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(interrupts=())

        async def astream(self, graph_input, config, stream_mode, **kwargs):
            assert graph_input == {
                "messages": [streaming.HumanMessage(content="покажи последние продажи")],
                "sql_result_table_blocks": [],
                "model_response_layout": None,
            }
            yield (
                "values",
                {
                    "__interrupt__": (
                        Interrupt(
                            value={
                                "action_requests": [
                                    {
                                        "name": "ask_user",
                                        "args": {"question": "За какой период показать продажи?"},
                                    }
                                ]
                            }
                        ),
                    )
                },
            )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "покажи последние продажи",
            "robot-hitl",
            "thread-hitl",
            queue,
            model_name="gpt-5.5",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    assert asyncio.run(exercise()) == [
        {"kind": "token", "text": "За какой период показать продажи?"}
    ]


def test_produce_agent_stream_async_resumes_pending_ask_user_interrupt(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")
    reset_checkpointer()

    class ToolCapableFakeChatModel(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    agent = build_chat_agent(
        ToolCapableFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"question": "За какой период показать продажи?"},
                            "id": "call-ask-user-1",
                        }
                    ],
                ),
                AIMessage(content="Показываю продажи за последние 30 дней."),
            ]
        ),
        [ai_tools.ask_user],
    )

    async def run_turn(user_text: str) -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            user_text,
            "robot-hitl-resume",
            "thread-hitl-resume",
            queue,
            model_name="gpt-5.5",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    first_items = asyncio.run(run_turn("покажи последние продажи"))
    second_items = asyncio.run(run_turn("за последние 30 дней"))

    assert {"kind": "token", "text": "За какой период показать продажи?"} in first_items
    assert {"kind": "token", "text": "Показываю продажи за последние 30 дней."} in second_items


def test_produce_agent_stream_async_real_agent_resumes_ask_user_with_streaming(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")
    reset_checkpointer()

    class ToolCapableFakeChatModel(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    agent = build_chat_agent(
        ToolCapableFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"question": "Между какими датами искать снижение?"},
                            "id": "call-ask-user-dates",
                        }
                    ],
                ),
                AIMessage(content="Продолжаю после ответа пользователя."),
            ]
        ),
        [ai_tools.ask_user],
    )

    async def run_turn(user_text: str) -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            user_text,
            "robot-real-hitl-resume",
            "thread-real-hitl-resume",
            queue,
            model_name="gpt-5.5",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    first_items = asyncio.run(run_turn("Найди причины снижения продаж"))
    second_items = asyncio.run(run_turn("реши сам"))

    assert {"kind": "token", "text": "Между какими датами искать снижение?"} in first_items
    assert {"kind": "token", "text": "Продолжаю после ответа пользователя."} in second_items


def test_produce_agent_stream_async_real_agent_resumes_after_prior_answer(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")
    reset_checkpointer()

    agent = build_chat_agent(
        ToolReadyFakeModel(
            responses=[
                AIMessage(content="Старый структурированный ответ."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"question": "Между какими датами искать снижение?"},
                            "id": "call-ask-user-after-structured",
                        }
                    ],
                ),
                AIMessage(content="Продолжаю анализ после уточнения дат."),
            ]
        ),
        [ai_tools.ask_user],
    )

    async def run_turn(user_text: str) -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            user_text,
            "robot-structured-hitl-resume",
            "thread-structured-hitl-resume",
            queue,
            model_name="gpt-5.5",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    first_items = asyncio.run(run_turn("Покажи динамику продаж"))
    second_items = asyncio.run(run_turn("Найди причины снижения"))
    third_items = asyncio.run(run_turn("между 07.06 и 11.06"))

    assert {"kind": "token", "text": "Старый структурированный ответ."} in first_items
    assert {"kind": "token", "text": "Между какими датами искать снижение?"} in second_items
    assert {"kind": "token", "text": "Продолжаю анализ после уточнения дат."} in third_items


def test_produce_agent_stream_async_resume_continuation_wins_over_stale_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {}

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    old_model_response_layout = {
        "blocks": [{"id": "old-c1", "type": "commentary", "content": "Старый анализ продаж."}]
    }

    class FakeAgent:
        def __init__(self) -> None:
            self.has_pending_question = False
            self.received_resume = False
            self.post_resume_continued = False

        async def aget_state(self, config):
            assert config["configurable"] == {"thread_id": "thread-hitl-lifecycle"}
            if not self.has_pending_question:
                return SimpleNamespace(interrupts=())
            return SimpleNamespace(
                values={"model_response_layout": old_model_response_layout, "sql_result_table_blocks": []},
                interrupts=(
                    Interrupt(
                        value={
                            "action_requests": [
                                {
                                    "name": "ask_user",
                                    "args": {"question": "В каком разрезе искать причину снижения?"},
                                }
                            ]
                        }
                    ),
                )
            )

        async def astream(self, graph_input, config, stream_mode, **kwargs):
            assert config["configurable"] == {"thread_id": "thread-hitl-lifecycle"}
            assert stream_mode == ["messages", "custom", "values"]
            if not isinstance(graph_input, Command):
                assert graph_input == {
                    "messages": [streaming.HumanMessage(content="Найди причины такого снижения по сумме продаж")],
                    "sql_result_table_blocks": [],
                    "model_response_layout": None,
                }
                self.has_pending_question = True
                yield (
                    "values",
                    {
                        "__interrupt__": (
                            Interrupt(
                                value={
                                    "action_requests": [
                                        {
                                            "name": "ask_user",
                                            "args": {"question": "В каком разрезе искать причину снижения?"},
                                        }
                                    ]
                                }
                            ),
                        )
                    },
                )
                return

            self.received_resume = True
            self.has_pending_question = False
            assert graph_input.resume == {
                "decisions": [{"type": "respond", "message": "реши сам, я хочу понять причины снижения продаж."}]
            }
            yield "values", {"model_response_layout": old_model_response_layout, "sql_result_table_blocks": []}
            self.post_resume_continued = True
            yield "messages", (streaming.AIMessageChunk(content="Продолжаю анализ после уточнения."), {})
            yield "values", {"messages": [streaming.AIMessage(content="Продолжаю анализ после уточнения.")]}

    agent = FakeAgent()

    async def run_turn(user_text: str) -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            user_text,
            "robot-hitl-lifecycle",
            "thread-hitl-lifecycle",
            queue,
            model_name="gpt-5.5",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    first_items = asyncio.run(run_turn("Найди причины такого снижения по сумме продаж"))
    second_items = asyncio.run(run_turn("реши сам, я хочу понять причины снижения продаж."))

    assert {"kind": "token", "text": "В каком разрезе искать причину снижения?"} in first_items
    assert agent.received_resume is True
    assert agent.post_resume_continued is True
    assert {"kind": "token", "text": "Продолжаю анализ после уточнения."} in second_items
    assert not any(isinstance(item, dict) and item.get("kind") == "blocks" for item in second_items)


def test_produce_agent_stream_async_streams_json_text_as_plain_text(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {}

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(interrupts=())

        async def astream(self, graph_input, config, stream_mode, **kwargs):
            old_json_suffix = '{"id":"old-c1","type":"commentary","content":"Старый анализ продаж."}]}'
            yield "messages", (streaming.AIMessageChunk(content='{"blocks":['), {})
            yield "messages", (streaming.AIMessageChunk(content=old_json_suffix), {})
            yield "values", {
                "messages": [
                    streaming.AIMessage(
                        content='{"blocks":[{"id":"old-c1","type":"commentary","content":"Старый анализ продаж."}]}'
                    )
                ]
            }

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "реши сам, я хочу понять причины снижения продаж.",
            "robot-stale-json",
            "thread-stale-json",
            queue,
            model_name="gpt-5.5",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {"kind": "token", "text": '{"blocks":['} in items
    assert {
        "kind": "token",
        "text": '{"id":"old-c1","type":"commentary","content":"Старый анализ продаж."}]}',
    } in items
    assert not any(isinstance(item, dict) and item.get("kind") == "blocks" for item in items)


def test_produce_agent_stream_async_suppresses_stale_final_ai_json_text(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {}

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    stale_json = '{"blocks":[{"id":"old-c1","type":"commentary","content":"Старый анализ продаж."}]}'
    stale_messages = [streaming.HumanMessage(content="старый вопрос"), streaming.AIMessage(content=stale_json)]

    class FakeAgent:
        async def aget_state(self, config):
            return SimpleNamespace(values={"messages": stale_messages}, interrupts=())

        async def astream(self, graph_input, config, stream_mode, **kwargs):
            yield "values", {"messages": stale_messages}

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "реши сам, я хочу понять причины снижения продаж.",
            "robot-stale-final-json",
            "thread-stale-final-json",
            queue,
            model_name="gpt-5.5",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert not any(isinstance(item, dict) and item.get("kind") == "token" for item in items)
    assert not any(isinstance(item, dict) and item.get("kind") == "blocks" for item in items)


def test_produce_agent_stream_async_uses_explicit_thread_id_not_robot_id(
    monkeypatch,
) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")
    captured: dict[str, Any] = {}

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            captured["graph_input"] = graph_input
            captured["config"] = config
            captured["stream_mode"] = stream_mode
            yield "messages", (streaming.AIMessageChunk(content="ok"), {})
            yield "values", {"messages": [streaming.AIMessage(content="ok")]}

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-dom-123",
            "thread-stable-456",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert captured["graph_input"] == {
        "messages": [streaming.HumanMessage(content="hello")],
        "sql_result_table_blocks": [],
        "model_response_layout": None,
    }
    assert captured["config"]["configurable"] == {"thread_id": "thread-stable-456"}
    assert captured["config"]["configurable"] != {"thread_id": "robot-dom-123"}
    assert captured["stream_mode"] == ["messages", "custom", "values"]
    assert {"kind": "token", "text": "ok"} in items


def test_produce_agent_stream_async_falls_back_to_final_ai_message(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            assert graph_input == {
                "messages": [streaming.HumanMessage(content="hello")],
                "sql_result_table_blocks": [],
                "model_response_layout": None,
            }
            assert config["configurable"] == {"thread_id": "thread-async-1"}
            assert stream_mode[:2] == ["messages", "custom"]
            assert "values" in stream_mode
            yield (
                "values",
                {"messages": [streaming.AIMessage(content="Привет! Чем помочь?")]},
            )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-async-1",
            "thread-async-1",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {"kind": "token", "text": "Привет! Чем помочь?"} in items


def test_produce_agent_stream_async_appends_missing_suffix_when_final_text_extends_stream(
    monkeypatch,
) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            assert graph_input == {
                "messages": [streaming.HumanMessage(content="hello")],
                "sql_result_table_blocks": [],
                "model_response_layout": None,
            }
            assert config["configurable"] == {"thread_id": "thread-async-2"}
            assert stream_mode[:2] == ["messages", "custom"]
            assert "values" in stream_mode
            yield "messages", (streaming.AIMessageChunk(content="Прив"), {})
            yield "values", {"messages": [streaming.AIMessage(content="Привет!")]}

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-async-2",
            "thread-async-2",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())
    token_items = [
        item for item in items if isinstance(item, dict) and item.get("kind") == "token"
    ]

    assert token_items == [
        {"kind": "token", "text": "Прив"},
        {"kind": "token", "text": "ет!"},
    ]


def test_produce_agent_stream_async_falls_back_to_text_block_content(
    monkeypatch,
) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            assert graph_input == {
                "messages": [streaming.HumanMessage(content="hello")],
                "sql_result_table_blocks": [],
                "model_response_layout": None,
            }
            assert config["configurable"] == {"thread_id": "thread-async-3"}
            assert stream_mode[:2] == ["messages", "custom"]
            assert "values" in stream_mode
            yield (
                "values",
                {
                    "messages": [
                        streaming.AIMessage(
                            content=[{"type": "text", "text": "Привет из блока!"}]
                        )
                    ]
                },
            )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-async-3",
            "thread-async-3",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {"kind": "token", "text": "Привет из блока!"} in items


def test_produce_agent_stream_async_ignores_malformed_model_response_layout(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            yield (
                "values",
                {
                    "model_response_layout": {"blocks": [{"type": "unknown"}]},
                    "messages": [streaming.AIMessage(content="Fallback markdown **works**")],
                },
            )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-async-malformed-structured",
            "thread-async-malformed-structured",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {"kind": "token", "text": "Fallback markdown **works**"} in items
    assert not any(isinstance(item, dict) and item.get("kind") == "blocks" for item in items)


def test_produce_agent_stream_async_renders_model_response_layout(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            yield (
                "values",
                {
                    "model_response_layout": {
                        "blocks": [{"id": "c1", "type": "commentary", "content": "Готово"}]
                    }
                },
            )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-structured-json",
            "thread-structured-json",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {
        "kind": "blocks",
        "blocks": [{"id": "c1", "type": "commentary", "format": "markdown", "content": "Готово"}],
    } in items


def test_produce_agent_stream_async_streams_pretty_json_text_as_plain_text(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            yield "messages", (streaming.AIMessageChunk(content="{\n  "), {})
            yield "messages", (streaming.AIMessageChunk(content='"blocks": [\n'), {})
            yield "messages", (
                streaming.AIMessageChunk(content='{"id":"c1","type":"commentary","content":"Готово"}\n]}'),
                {},
            )
            yield (
                "values",
                {
                    "messages": [
                        streaming.AIMessage(
                            content='{\n  "blocks": [\n{"id":"c1","type":"commentary","content":"Готово"}\n]}'
                        )
                    ]
                },
            )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-pretty-structured-json",
            "thread-pretty-structured-json",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert {"kind": "token", "text": "{\n  "} in items
    assert {"kind": "token", "text": '"blocks": [\n'} in items
    assert {"kind": "token", "text": '{"id":"c1","type":"commentary","content":"Готово"}\n]}'} in items
    assert not any(isinstance(item, dict) and item.get("kind") == "blocks" for item in items)


def test_produce_agent_stream_async_keeps_semicolon_tokens_unchanged(
    monkeypatch,
) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")

    class FakeUsageMetadataCallbackHandler:
        def __init__(self) -> None:
            self.usage_metadata = {
                "gpt-5.4": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
            }

    monkeypatch.setattr(
        streaming,
        "UsageMetadataCallbackHandler",
        FakeUsageMetadataCallbackHandler,
    )

    class FakeAgent:
        async def astream(self, graph_input, config, stream_mode, **kwargs):
            assert graph_input == {
                "messages": [streaming.HumanMessage(content="hello")],
                "sql_result_table_blocks": [],
                "model_response_layout": None,
            }
            assert config["configurable"] == {"thread_id": "thread-async-4"}
            assert stream_mode[:2] == ["messages", "custom"]
            assert "values" in stream_mode
            yield "messages", (streaming.AIMessageChunk(content="SELECT 1;"), {})
            yield (
                "values",
                {"messages": [streaming.AIMessage(content="SELECT 1; next")]},
            )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            FakeAgent(),
            "hello",
            "robot-async-4",
            "thread-async-4",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())
    token_items = [
        item for item in items if isinstance(item, dict) and item.get("kind") == "token"
    ]

    assert token_items == [
        {"kind": "token", "text": "SELECT 1;"},
        {"kind": "token", "text": " next"},
    ]


class ToolReadyFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def test_produce_agent_stream_async_runs_memory_middleware_before_each_model_call(
    monkeypatch,
) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")
    reset_checkpointer()
    before_model_calls: list[int] = []
    original_before_model = type(trim_chat_history_middleware).before_model

    def counting_before_model(self, state, runtime):
        before_model_calls.append(len(state.get("messages", [])))
        return original_before_model(self, state, runtime)

    monkeypatch.setattr(
        type(trim_chat_history_middleware), "before_model", counting_before_model
    )

    @tool
    def lookup(q: str) -> str:
        """Return a deterministic lookup result."""

        return f"result:{q}"

    agent = build_chat_agent(
        ToolReadyFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call-1", "name": "lookup", "args": {"q": "abc"}}
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        [lookup],
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            "hello",
            "robot-phase3-stream",
            "phase3-stream-thread",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert before_model_calls == [1, 3]
    assert {"kind": "token", "text": "done"} in items


def test_produce_agent_stream_async_preserves_checkpointed_validated_queries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(streaming, "get_api_key", lambda: "test-key")
    reset_checkpointer()

    agent = build_chat_agent(
        ToolReadyFakeModel(
            responses=[AIMessage(content="first"), AIMessage(content="second")]
        ),
        [],
    )
    config = cast(Any, {"configurable": {"thread_id": "phase-final-validated-thread"}})

    agent.invoke(
        cast(
            Any,
            {
                "messages": [{"role": "user", "content": "hello"}],
                "validated_queries": {
                    "token-1": {
                        "sql": "SELECT 1",
                        "source_id": "demo",
                        "dialect": "sqlite",
                        "created_at": 1.0,
                        "expires_at": 2.0,
                        "thread_id": "phase-final-validated-thread",
                        "status": "validated",
                    }
                },
            },
        ),
        config=config,
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        await streaming.produce_agent_stream_async(
            agent,
            "again",
            "robot-preserve-1",
            "phase-final-validated-thread",
            queue,
            model_name="gpt-5.4",
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    asyncio.run(exercise())

    state = agent.get_state(config).values
    assert state["validated_queries"]["token-1"]["sql"] == "SELECT 1"
