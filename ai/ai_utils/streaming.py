"""Streaming helpers for usage aggregation and async agent output pumping."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from loguru import logger
from pydantic import ValidationError

from .logging_config import build_log_preview, build_short_log_id, format_log_event
from .runtime_config import get_api_key
from .structured_output_blocks import (
    AgentCommentaryResponse,
    AgentFinalResponse,
    CommentaryBlock,
    DataTableBlock,
    DataTableDetails,
)
from .token_usage import calculate_openai_usage_cost

ASK_USER_TOOL_NAME = "ask_user"


def extract_final_ai_text(data: Any) -> str:
    """Return the latest non-tool AI message text from a streamed graph state.

    Example:
        >>> extract_final_ai_text({"messages": [AIMessage(content="hello")]})
        'hello'
    """
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        final_text = extract_ai_message_text(message)
        if final_text:
            return final_text
    return ""


def extract_state_value(data: Any) -> dict[str, Any]:
    """Return a graph state dictionary from streamed values or checkpoint wrappers."""

    if isinstance(data, dict):
        return data
    values = getattr(data, "values", None)
    if isinstance(values, dict):
        return values
    value = getattr(data, "value", None)
    if isinstance(value, dict):
        return value
    return {}


def extract_ai_message_text(message: AIMessage) -> str:
    """Extract plain text from a final AI message content payload.

    Example:
        >>> extract_ai_message_text(AIMessage(content=[{"type": "text", "text": "hello"}]))
        'hello'
    """
    content = message.content
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text") or "")
            if text:
                text_parts.append(text)
    return "".join(text_parts)


def extract_model_response_layout(data: Any) -> AgentCommentaryResponse | None:
    """Return a validated app-owned model response layout from graph state when present."""

    value = extract_state_value(data)
    if "model_response_layout" not in value:
        return None
    model_response_layout = value["model_response_layout"]
    if model_response_layout is None:
        return None
    if isinstance(model_response_layout, AgentCommentaryResponse):
        return model_response_layout
    try:
        return AgentCommentaryResponse.model_validate(model_response_layout)
    except ValidationError as exc:
        logger.warning(format_log_event("tool.ui", "layout_invalid", error=build_log_preview(exc)))
        return None


def build_final_response_blocks(data: Any, response: AgentCommentaryResponse) -> list[dict[str, Any]]:
    """Resolve LLM table placeholders against backend-owned SQL result blocks."""

    state = extract_state_value(data)
    sql_result_table_blocks: list[DataTableBlock] = []
    for index, block in enumerate(state.get("sql_result_table_blocks") or []):
        try:
            sql_result_table_blocks.append(DataTableBlock.model_validate(block))
        except ValidationError as exc:
            logger.warning(
                format_log_event(
                    "tool.ui",
                    "sql_block_malformed",
                    index=index,
                    error=build_log_preview(exc),
                )
            )
    resolved_blocks: list[CommentaryBlock | DataTableBlock] = []
    next_table_index = 0
    for block in response.blocks:
        if block.type == "commentary":
            resolved_blocks.append(block)
            continue
        if next_table_index < len(sql_result_table_blocks):
            table_block = sql_result_table_blocks[next_table_index]
            if block.title and not table_block.title:
                table_block = table_block.model_copy(update={"title": block.title})
            if block.notes:
                details_data = table_block.details.model_dump() if table_block.details else {}
                table_block = table_block.model_copy(
                    update={"details": DataTableDetails(**{**details_data, "notes": block.notes})}
                )
            resolved_blocks.append(table_block)
            next_table_index += 1
            continue
        logger.warning(
            format_log_event(
                "tool.ui",
                "placeholder_unmatched",
                id=block.id,
                title=block.title or "",
                available_sql_blocks=len(sql_result_table_blocks),
            )
        )
    resolved_blocks.extend(sql_result_table_blocks[next_table_index:])
    return AgentFinalResponse(blocks=resolved_blocks).model_dump(exclude_none=True)["blocks"]


def count_state_messages(data: Any) -> int:
    """Return the number of messages in a graph state wrapper."""

    messages = extract_state_value(data).get("messages")
    return len(messages) if isinstance(messages, list) else 0


def log_new_state_messages(data: Any, logged_message_count: int) -> int:
    """Log newly observed graph messages that affect final-answer routing."""

    messages = extract_state_value(data).get("messages")
    if not isinstance(messages, list):
        return logged_message_count
    for index, message in enumerate(messages[logged_message_count:], start=logged_message_count):
        if isinstance(message, AIMessage):
            tool_calls = message.tool_calls or []
            if tool_calls:
                logger.debug(
                    format_log_event(
                        "chat.message",
                        "ai_requested_tools",
                        index=index,
                        count=len(tool_calls),
                        detail_lines=[
                            _format_tool_call_detail(position, tool_call)
                            for position, tool_call in enumerate(tool_calls, start=1)
                        ],
                    )
                )
            else:
                logger.debug(
                    format_log_event(
                        "chat.message",
                        "final_text",
                        index=index,
                        len=len(extract_ai_message_text(message)),
                    )
                )
        elif isinstance(message, ToolMessage):
            content = str(message.content or "")
            logger.debug(
                format_log_event(
                    "chat.message",
                    "tool_result",
                    index=index,
                    tool=message.name,
                    len=len(content),
                    preview=build_log_preview(content[:300]) if message.name == "submit_model_response_layout" else "",
                )
            )
    return len(messages)


def extract_latest_fresh_ai_text_preview(data: Any, baseline_message_count: int, limit: int = 500) -> str:
    """Return a preview of the latest fresh non-tool AI text for diagnostics."""

    text = extract_fresh_final_ai_text(data, baseline_message_count)
    return text[:limit]


def _summarize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {"type": type(tool_call).__name__}
    args = tool_call.get("args")
    summary: dict[str, Any] = {
        "name": tool_call.get("name"),
        "id": tool_call.get("id"),
        "args_keys": sorted(args.keys()) if isinstance(args, dict) else None,
    }
    if tool_call.get("name") == "submit_model_response_layout" and isinstance(args, dict):
        layout = args.get("layout")
        blocks = layout.get("blocks") if isinstance(layout, dict) else None
        summary["layout_type"] = type(layout).__name__
        summary["layout_block_count"] = len(blocks) if isinstance(blocks, list) else None
    return summary


def _format_tool_call_detail(position: int, tool_call: Any) -> str:
    summary = _summarize_tool_call(tool_call)
    name = str(summary.get("name") or summary.get("type") or "unknown")
    args = summary.get("args_keys")
    call_id = summary.get("id")
    parts = [f"{position}. {name}"]
    if isinstance(args, list):
        parts.append(f"args={','.join(str(arg) for arg in args)}")
    if call_id:
        parts.append(f"call={build_short_log_id(call_id, length=12)}")
    layout_block_count = summary.get("layout_block_count")
    if layout_block_count is not None:
        parts.append(f"layout_blocks={layout_block_count}")
    return " ".join(parts)


def count_sql_result_blocks(data: Any) -> int:
    """Return the number of SQL result blocks in a graph state wrapper."""

    sql_result_table_blocks = extract_state_value(data).get("sql_result_table_blocks")
    return len(sql_result_table_blocks) if isinstance(sql_result_table_blocks, list) else 0


def has_fresh_final_ai_message(data: Any, baseline_message_count: int) -> bool:
    """Return whether state contains a non-tool AI message appended during this run."""

    messages = extract_state_value(data).get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages[baseline_message_count:]:
        if isinstance(message, AIMessage) and not message.tool_calls and extract_ai_message_text(message):
            return True
    return False


def extract_fresh_final_ai_text(data: Any, baseline_message_count: int) -> str:
    """Return latest non-tool AI message text appended during this run."""

    messages = extract_state_value(data).get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages[baseline_message_count:]):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        final_text = extract_ai_message_text(message)
        if final_text:
            return final_text
    return ""


def with_fresh_sql_result_blocks(data: Any, baseline_sql_result_block_count: int) -> dict[str, Any]:
    """Return a render state that only exposes SQL result blocks produced during this run."""

    state = dict(extract_state_value(data))
    sql_result_table_blocks = state.get("sql_result_table_blocks")
    if isinstance(sql_result_table_blocks, list):
        state["sql_result_table_blocks"] = sql_result_table_blocks[baseline_sql_result_block_count:]
    return state


def is_current_run_model_response_layout(
    data: Any,
    model_response_layout: AgentCommentaryResponse,
    baseline_model_response_layout: AgentCommentaryResponse | None,
    baseline_message_count: int,
    baseline_sql_result_block_count: int,
) -> bool:
    """Return whether a model response layout belongs to the current graph run."""

    if model_response_layout != baseline_model_response_layout:
        return True
    return has_fresh_final_ai_message(data, baseline_message_count) and (
        count_sql_result_blocks(data) > baseline_sql_result_block_count
    )


async def get_checkpoint_state_value(agent: Any, config: RunnableConfig) -> dict[str, Any]:
    """Return the current checkpoint state values for agents that expose state."""

    if not hasattr(agent, "aget_state"):
        return {}
    return extract_state_value(await agent.aget_state(config))


def unpack_stream_chunk(chunk: Any) -> tuple[str, Any] | None:
    """Return `(mode, data)` from LangGraph stream chunks with or without subgraph namespace."""

    if isinstance(chunk, dict):
        mode = chunk.get("type")
        if not isinstance(mode, str):
            return None
        data = chunk.get("data")
        if mode == "values" and isinstance(data, dict) and chunk.get("interrupts"):
            data = {**data, "__interrupt__": chunk.get("interrupts")}
        return mode, data
    if not isinstance(chunk, tuple):
        return None
    if len(chunk) == 2:
        mode, data = chunk
        return (mode, data) if isinstance(mode, str) else None
    if len(chunk) == 3:
        _namespace, mode, data = chunk
        return (mode, data) if isinstance(mode, str) else None
    return None


def build_final_text_delta(streamed_text: str, final_text: str) -> str:
    """Return the missing suffix needed to complete streamed assistant text.

    Example:
        >>> build_final_text_delta("Прив", "Привет!")
        'ет!'
    """
    if not final_text:
        return ""
    if not streamed_text:
        return final_text
    if final_text.startswith(streamed_text):
        return final_text[len(streamed_text) :]
    return ""


def build_usage_event(
    usage_by_model: Mapping[str, Mapping[str, Any]], configured_model_name: str
) -> dict[str, Any] | None:
    """Build one queue event with request-scoped usage totals.

    Example:
        >>> build_usage_event(
        ...     {"gpt-5.4": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        ...     "gpt-5.4",
        ... )
        {
            'kind': 'usage',
            'model': 'gpt-5.4',
            'usage_metadata': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
        }

    The emitted payload intentionally mirrors LangChain/LangSmith-compatible
    ``usage_metadata`` so later phases can reuse it for pricing and UI updates.
    """
    if not usage_by_model:
        return None

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    input_token_details: dict[str, int] = {}
    output_token_details: dict[str, int] = {}
    model_names = [model_name for model_name in usage_by_model if model_name]

    for usage_metadata in usage_by_model.values():
        total_input_tokens += int(usage_metadata.get("input_tokens") or 0)
        total_output_tokens += int(usage_metadata.get("output_tokens") or 0)
        total_tokens += int(usage_metadata.get("total_tokens") or 0)
        _merge_token_details(
            input_token_details, usage_metadata.get("input_token_details")
        )
        _merge_token_details(
            output_token_details, usage_metadata.get("output_token_details")
        )

    payload_usage_metadata: dict[str, Any] = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
    }
    if input_token_details:
        payload_usage_metadata["input_token_details"] = input_token_details
    if output_token_details:
        payload_usage_metadata["output_token_details"] = output_token_details

    payload_model = configured_model_name
    if len(model_names) == 1:
        payload_model = model_names[0]

    cost_usd = calculate_openai_usage_cost(
        configured_model_name, payload_usage_metadata
    )

    payload = {
        "kind": "usage",
        "model": payload_model,
        "pricing_model": configured_model_name,
        "usage_metadata": payload_usage_metadata,
    }
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    return payload


def extract_ask_user_questions_from_interrupts(interrupts: Any) -> list[str]:
    """Return pending `ask_user` questions from LangGraph interrupt payloads."""
    questions: list[str] = []
    for interrupt in interrupts or ():
        interrupt_payload = getattr(interrupt, "value", None)
        if not isinstance(interrupt_payload, dict):
            continue
        for action in interrupt_payload.get("action_requests") or []:
            if not isinstance(action, dict) or action.get("name") != ASK_USER_TOOL_NAME:
                continue
            action_args = action.get("args") or action.get("arguments") or {}
            if not isinstance(action_args, dict):
                continue
            question = str(action_args.get("question") or "").strip()
            if question:
                questions.append(question)
    return questions


def build_ask_user_message(questions: list[str]) -> str:
    """Join one or more clarification questions for the existing single-input UI."""
    return "\n\n".join(question for question in questions if question.strip())


def build_ask_user_resume_command(user_text: str, decision_count: int) -> Command:
    """Build a respond decision for each pending `ask_user` action."""
    logger.debug(
        format_log_event(
            "chat.resume",
            "command_built",
            decisions=decision_count,
            user_len=len(user_text),
        )
    )
    return Command(
        resume={
            "decisions": [
                {"type": "respond", "message": user_text} for _index in range(decision_count)
            ]
        }
    )


async def get_checkpointed_ask_user_questions(agent: Any, config: RunnableConfig) -> list[str]:
    """Return pending `ask_user` questions saved in the graph checkpoint."""
    if not hasattr(agent, "aget_state"):
        logger.debug(format_log_event("chat.graph", "checkpoint_missing", reason="no_aget_state"))
        return []
    state = await agent.aget_state(config)
    interrupts = getattr(state, "interrupts", ())
    questions = extract_ask_user_questions_from_interrupts(interrupts)
    logger.debug(
        format_log_event(
            "chat.graph",
            "checkpoint_found",
            interrupts=len(interrupts or ()),
            questions=len(questions),
        )
    )
    return questions


async def build_agent_graph_input(agent: Any, user_text: str, config: RunnableConfig) -> dict[str, Any] | Command:
    """Return either a fresh user message input or a HITL resume command."""
    if not hasattr(agent, "aget_state"):
        logger.debug(format_log_event("chat.graph", "checkpoint_missing", reason="no_aget_state"))
        logger.debug(format_log_event("chat.graph", "input_ready", mode="fresh_user_message"))
        return {
            "messages": [HumanMessage(content=user_text)],
            "sql_result_table_blocks": [],
            "model_response_layout": None,
        }

    state = await agent.aget_state(config)
    interrupts = getattr(state, "interrupts", ())
    questions = extract_ask_user_questions_from_interrupts(interrupts)
    logger.debug(
        format_log_event(
            "chat.graph",
            "checkpoint_found",
            interrupts=len(interrupts or ()),
            questions=len(questions),
        )
    )
    if questions:
        logger.debug(
            format_log_event(
                "chat.graph",
                "input_ready",
                mode="hitl_resume",
                pending_questions=len(questions),
            )
        )
        return build_ask_user_resume_command(user_text, len(questions))
    logger.debug(format_log_event("chat.graph", "input_ready", mode="fresh_user_message"))
    return {"messages": [HumanMessage(content=user_text)], "sql_result_table_blocks": [], "model_response_layout": None}


async def emit_usage_event_async(
    callback: UsageMetadataCallbackHandler,
    configured_model_name: str,
    queue: asyncio.Queue[Any],
) -> None:
    """Enqueue one usage event after an async graph run completes."""
    usage_event = build_usage_event(callback.usage_metadata, configured_model_name)
    if usage_event is None:
        logger.debug(format_log_event("chat.usage", "missing", model=configured_model_name))
        return

    _log_usage_event(usage_event)
    await queue.put(usage_event)


def _log_usage_event(usage_event: Mapping[str, Any]) -> None:
    """Log a usage payload before sending it to the UI queue."""

    usage_metadata = usage_event["usage_metadata"]
    input_token_details = usage_metadata.get("input_token_details")
    cache_read_tokens = 0
    if isinstance(input_token_details, dict):
        cache_read_tokens = int(input_token_details.get("cache_read") or 0)
    logger.debug(
        format_log_event(
            "chat.usage",
            "raw_received",
            model=usage_event["model"],
            in_=usage_metadata.get("input_tokens", 0),
            out=usage_metadata.get("output_tokens", 0),
            total=usage_metadata.get("total_tokens", 0),
            cache=cache_read_tokens,
        )
    )
    logger.info(
        format_log_event(
            "chat.usage",
            "recorded",
            model=usage_event["model"],
            in_=usage_metadata.get("input_tokens", 0),
            out=usage_metadata.get("output_tokens", 0),
            total=usage_metadata.get("total_tokens", 0),
            cache=cache_read_tokens,
            cost=_format_usage_cost(usage_event.get("cost_usd")),
        )
    )


def _format_usage_cost(cost_usd: object) -> str | None:
    if cost_usd is None:
        return None
    return f"${float(cost_usd):.4f}"


def _merge_token_details(target: dict[str, int], details: Any) -> None:
    """Merge token detail counters into the target mapping.

    Example:
        >>> details = {"cache_read": 4}
        >>> target = {"cache_read": 1}
        >>> _merge_token_details(target, details)
        >>> target
        {'cache_read': 5}
    """
    if not isinstance(details, dict):
        return
    for key, value in details.items():
        target[str(key)] = target.get(str(key), 0) + int(value or 0)


async def produce_agent_stream_async(
    agent: Any,
    user_text: str,
    robot_id: str,
    thread_id: str,
    queue: asyncio.Queue[Any],
    *,
    model_name: str,
    enable_streaming: bool = True,
) -> None:
    """Run the graph asynchronously and put assistant events into the queue."""
    if not get_api_key():
        raise ValueError("OPENAI_API_KEY is not set")

    usage_callback = UsageMetadataCallbackHandler()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [usage_callback],
        "recursion_limit": 30,
    }
    graph_input = await build_agent_graph_input(agent, user_text, config)
    baseline_state = await get_checkpoint_state_value(agent, config)
    baseline_message_count = count_state_messages(baseline_state)
    baseline_model_response_layout = extract_model_response_layout(baseline_state)
    baseline_sql_result_block_count = count_sql_result_blocks(baseline_state)
    if not isinstance(graph_input, Command):
        baseline_model_response_layout = None
        baseline_sql_result_block_count = 0
    logger.debug(
        format_log_event(
            "chat.graph",
            "baseline_ready",
            messages=baseline_message_count,
            layout=baseline_model_response_layout is not None,
            sql_blocks=baseline_sql_result_block_count,
        )
    )
    if isinstance(graph_input, Command):
        resume_payload = getattr(graph_input, "resume", None)
        decisions = resume_payload.get("decisions") if isinstance(resume_payload, dict) else None
        logger.debug(
            format_log_event(
                "chat.resume",
                "confirmed",
                type=type(resume_payload).__name__,
                decisions=len(decisions) if isinstance(decisions, list) else 0,
            )
        )
    else:
        logger.debug(format_log_event("chat.graph", "input_confirmed", mode="fresh_user_message"))
    logger.debug(
        format_log_event(
            "chat.agent",
            "started",
            thread=build_short_log_id(thread_id),
            streaming=enable_streaming,
        )
    )

    if enable_streaming:
        visible_streamed_text = ""
        final_text = ""
        model_response_layout: AgentCommentaryResponse | None = None
        response_blocks: list[dict[str, Any]] | None = None
        interrupted_questions: list[str] = []
        logged_message_count = baseline_message_count
        async for chunk in agent.astream(
            graph_input,
            config=config,
            stream_mode=["messages", "custom", "values"],
            version="v2",
        ):
            stream_chunk = unpack_stream_chunk(chunk)
            if stream_chunk is None:
                logger.debug(format_log_event("chat.graph", "chunk_ignored", chunk_type=type(chunk).__name__))
                continue
            mode, data = stream_chunk
            if mode == "values":
                state = extract_state_value(data)
                interrupts = getattr(data, "interrupts", ())
                if isinstance(data, dict):
                    interrupts = data.get("__interrupt__") or data.get("interrupts") or interrupts
                logger.debug(
                    format_log_event(
                        "chat.graph",
                        "state_received",
                        messages=count_state_messages(state),
                        layout=extract_model_response_layout(state) is not None,
                        sql_blocks=count_sql_result_blocks(state),
                        interrupts=len(interrupts or ()),
                    )
                )
                logged_message_count = log_new_state_messages(state, logged_message_count)
            if mode == "messages":
                token, _metadata = data
                if isinstance(token, AIMessageChunk) and token.text:
                    token_text = token.text
                    visible_streamed_text += token_text
                    await queue.put({"kind": "token", "text": token_text})
            elif mode == "custom":
                if isinstance(data, dict) and data.get("kind") == "theme":
                    theme = str(data.get("theme") or "").strip()
                    if theme:
                        logger.debug(format_log_event("tool.ui", "theme_switch", theme=theme))
                        await queue.put({"kind": "theme", "theme": theme})
                elif isinstance(data, str) and data.strip():
                    await queue.put({"kind": "progress", "text": data.strip()})
            elif mode == "values":
                questions = extract_ask_user_questions_from_interrupts(getattr(data, "interrupts", ()))
                if not questions and isinstance(data, dict):
                    questions = extract_ask_user_questions_from_interrupts(data.get("__interrupt__"))
                if questions:
                    interrupted_questions = questions
                    break
                current_model_response_layout = extract_model_response_layout(data)
                if current_model_response_layout is not None:
                    if is_current_run_model_response_layout(
                        data,
                        current_model_response_layout,
                        baseline_model_response_layout,
                        baseline_message_count,
                        baseline_sql_result_block_count,
                    ):
                        model_response_layout = current_model_response_layout
                        render_state = with_fresh_sql_result_blocks(data, baseline_sql_result_block_count)
                        response_blocks = build_final_response_blocks(render_state, model_response_layout)
                        logger.debug(format_log_event("tool.ui", "layout_accepted"))
                    else:
                        logger.debug(format_log_event("tool.ui", "layout_rejected", reason="stale_checkpoint"))
                fresh_final_text = extract_fresh_final_ai_text(data, baseline_message_count)
                if fresh_final_text:
                    final_text = fresh_final_text
                elif extract_final_ai_text(data):
                    logger.debug(format_log_event("chat.message", "final_text_rejected", reason="not_fresh"))
        if not interrupted_questions:
            interrupted_questions = await get_checkpointed_ask_user_questions(agent, config)
        if interrupted_questions:
            question_text = build_ask_user_message(interrupted_questions)
            if question_text:
                await queue.put({"kind": "token", "text": question_text})
            await emit_usage_event_async(usage_callback, model_name, queue)
            logger.debug(format_log_event("chat.graph", "interrupted", thread=build_short_log_id(thread_id)))
            logger.debug(format_log_event("chat.request", "hitl_waiting", thread=build_short_log_id(thread_id)))
            return
        if model_response_layout is not None:
            logger.debug(format_log_event("tool.ui", "layout_ready", blocks=len(model_response_layout.blocks)))
            await queue.put({"kind": "blocks", "blocks": response_blocks or []})
        else:
            final_state = await get_checkpoint_state_value(agent, config)
            final_sql_result_block_count = count_sql_result_blocks(final_state)
            log = logger.warning if final_sql_result_block_count > baseline_sql_result_block_count else logger.debug
            log(
                format_log_event(
                    "tool.ui",
                    "layout_missing",
                    sql_blocks=final_sql_result_block_count,
                    baseline_sql_blocks=baseline_sql_result_block_count,
                    final_text_len=len(final_text),
                    streamed_text_len=len(visible_streamed_text),
                    messages=count_state_messages(final_state),
                    final_text_preview=build_log_preview(
                        extract_latest_fresh_ai_text_preview(final_state, baseline_message_count)
                    ),
                )
            )
            final_text_delta = build_final_text_delta(visible_streamed_text, final_text)
            if final_text_delta:
                await queue.put({"kind": "token", "text": final_text_delta})
        await emit_usage_event_async(usage_callback, model_name, queue)
        logger.debug(format_log_event("chat.agent", "completed", thread=build_short_log_id(thread_id), mode="stream"))
        return

    result = await agent.ainvoke(graph_input, config=config, version="v2")
    await emit_usage_event_async(usage_callback, model_name, queue)
    logger.debug(format_log_event("chat.agent", "completed", thread=build_short_log_id(thread_id), mode="invoke"))

    interrupted_questions = extract_ask_user_questions_from_interrupts(getattr(result, "interrupts", ()))
    if not interrupted_questions and isinstance(result, dict):
        interrupted_questions = extract_ask_user_questions_from_interrupts(result.get("__interrupt__"))
    if not interrupted_questions:
        interrupted_questions = await get_checkpointed_ask_user_questions(agent, config)
    if interrupted_questions:
        question_text = build_ask_user_message(interrupted_questions)
        if question_text:
            await queue.put({"kind": "token", "text": question_text})
        return

    result_value = extract_state_value(result)
    log_new_state_messages(result_value, baseline_message_count)
    model_response_layout = extract_model_response_layout(result_value)
    if model_response_layout is not None and is_current_run_model_response_layout(
        result_value,
        model_response_layout,
        baseline_model_response_layout,
        baseline_message_count,
        baseline_sql_result_block_count,
    ):
        logger.debug(format_log_event("tool.ui", "layout_ready", blocks=len(model_response_layout.blocks)))
        render_state = with_fresh_sql_result_blocks(result_value, baseline_sql_result_block_count)
        await queue.put({"kind": "blocks", "blocks": build_final_response_blocks(render_state, model_response_layout)})
        return
    messages_out = result_value.get("messages") if isinstance(result_value, dict) else None
    if isinstance(messages_out, list):
        for msg in reversed(messages_out[baseline_message_count:]):
            if not isinstance(msg, AIMessage) or msg.tool_calls:
                continue
            content = extract_ai_message_text(msg)
            if content:
                await queue.put({"kind": "token", "text": content})
                return
    await queue.put({"kind": "token", "text": ""})
