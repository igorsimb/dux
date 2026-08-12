"""Chat runtime orchestration for model/tool setup and chat execution."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain.chat_models import init_chat_model
from loguru import logger

from core.db_config.source_database_router import get_sql_database_for_source

from .sql_tools import (
    build_guarded_query_tool,
    clear_run_query_tools,
    load_allowed_source_candidates,
    set_run_query_tool_factory_for_source,
    set_run_query_tool_for_source,
)
from ..ai_tools import (
    ask_user,
    execute_validated_sql,
    get_table_descriptions,
    get_table_metadata,
    submit_model_response_layout,
    validate_sql,
)
from ..ai_tools_extra import switch_color_theme
from .chat_errors import build_error_message, build_user_facing_message
from .chat_agent import build_chat_agent
from .checkpointer import delete_thread_checkpoints
from .logging_config import build_public_conversation_code, build_short_log_id, format_log_event
from .runtime_config import get_model_name, get_openai_proxy
from .streaming import produce_agent_stream_async


def build_model_and_tools(
    stream_timeout_seconds: int,
) -> tuple[Any, list[Any], str]:
    """Build the streaming model and guarded tools list for one chat run."""
    source_candidates = load_allowed_source_candidates()
    if not source_candidates:
        raise ValueError("No allowlisted source+dialect candidates configured")

    source_ids: list[str] = []
    seen_source_ids: set[str] = set()
    for source_id, _dialect in source_candidates:
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        source_ids.append(source_id)

    openai_proxy = get_openai_proxy()
    model_name = get_model_name()
    model = init_chat_model(
        model=model_name,
        use_responses_api=True,
        openai_proxy=openai_proxy,
        streaming=True,
        stream_usage=True,
        request_timeout=stream_timeout_seconds,
        max_retries=1,
    )

    def build_run_query_tool_for_source(source_id: str) -> Any:
        db = get_sql_database_for_source(source_id)
        return build_guarded_query_tool(db)

    clear_run_query_tools()
    multiple_sources = len(source_ids) > 1
    for source_id in source_ids:
        if multiple_sources:
            set_run_query_tool_factory_for_source(
                source_id,
                lambda source_id=source_id: build_run_query_tool_for_source(source_id),
            )
            continue

        db = get_sql_database_for_source(source_id)
        run_query_tool = build_guarded_query_tool(db)
        set_run_query_tool_for_source(source_id, run_query_tool)

    tools = [
        ask_user,
        get_table_descriptions,
        get_table_metadata,
        switch_color_theme,
        validate_sql,
        execute_validated_sql,
        submit_model_response_layout,
    ]
    tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
    logger.debug(
        format_log_event(
            "chat.agent",
            "tools_ready",
            count=len(tool_names),
            detail_lines=[f"tools: {', '.join(tool_names)}"],
        )
    )
    return model, tools, model_name


def _is_missing_tool_response_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "assistant message with 'tool_calls' must be followed by tool messages"
        in message.lower()
        and "tool_call_id" in message.lower()
    )


async def _enqueue_error_token(
    queue: asyncio.Queue[Any | None], error_text: str, conversation_code: str
) -> None:
    await queue.put(
        {
            "kind": "token",
            "text": error_text,
            "display_text": build_user_facing_message(error_text, conversation_code),
        }
    )


async def _run_chat_session_once(
    *,
    user_text: str,
    robot_id: str,
    thread_id: str,
    queue: asyncio.Queue[Any | None],
    stream_timeout_seconds: int,
) -> None:
    model, tools, model_name = build_model_and_tools(stream_timeout_seconds)
    agent = build_chat_agent(model, tools)
    await produce_agent_stream_async(
        agent,
        user_text,
        robot_id,
        thread_id,
        queue,
        model_name=model_name,
    )


async def run_chat_session(
    *,
    user_text: str,
    robot_id: str,
    thread_id: str,
    queue: asyncio.Queue[Any | None],
    stream_timeout_seconds: int,
) -> None:
    """Run one async chat session and push UI events into the queue.

    Example:
        If the session raises a runtime error, the helper still enqueues a
        user-visible error token followed by the terminal ``None`` sentinel.
    """

    conversation_code = build_public_conversation_code(thread_id)

    try:
        await _run_chat_session_once(
            user_text=user_text,
            robot_id=robot_id,
            thread_id=thread_id,
            queue=queue,
            stream_timeout_seconds=stream_timeout_seconds,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if _is_missing_tool_response_error(exc):
            logger.warning(
                format_log_event(
                    "chat.graph",
                    "checkpoint_resetting",
                    thread=build_short_log_id(thread_id),
                    reason="unmatched_tool_call",
                )
            )
            try:
                delete_thread_checkpoints(thread_id)
            except Exception as reset_exc:
                logger.exception(
                    format_log_event("chat.graph", "checkpoint_reset_failed", thread=build_short_log_id(thread_id))
                )
                error_text = build_error_message(reset_exc)
                await _enqueue_error_token(queue, error_text, conversation_code)
                return
            try:
                await _run_chat_session_once(
                    user_text=user_text,
                    robot_id=robot_id,
                    thread_id=thread_id,
                    queue=queue,
                    stream_timeout_seconds=stream_timeout_seconds,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as retry_exc:
                logger.exception(
                    format_log_event("chat.agent", "failed_after_reset", thread=build_short_log_id(thread_id))
                )
                error_text = build_error_message(retry_exc)
                await _enqueue_error_token(queue, error_text, conversation_code)
                return
        logger.exception(format_log_event("chat.agent", "failed", thread=build_short_log_id(thread_id)))
        error_text = build_error_message(exc)
        await _enqueue_error_token(queue, error_text, conversation_code)
    finally:
        await queue.put(None)
