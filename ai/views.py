import asyncio
import uuid
from typing import AsyncGenerator, Any

from asgiref.sync import sync_to_async
from datastar_py.django import (
    DatastarResponse,
    ServerSentEventGenerator as SSE,
    read_signals,
)
from datastar_py.sse import DatastarEvent
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest
from django.shortcuts import render
from loguru import logger

from .ai_utils import (
    ROBOT_ID_PREFIX,
    append_robot_blocks,
    append_robot_container,
    append_robot_text,
    append_user_message,
    build_blocks_visible_text,
    build_error_message,
    build_timeout_message,
    build_user_facing_message,
    run_chat_session,
)
from .ai_utils.chat_logging import append_chat_log_turn
from .ai_utils.logging_config import (
    build_log_preview,
    build_public_conversation_code,
    build_short_log_id,
    format_log_event,
)
from .ai_utils.chat_session import build_thread_id, normalize_chat_session_key
from .ai_utils.runtime_config import get_model_name
from .ai_utils.token_usage import (
    format_cost_usd,
    format_compact_tokens,
    format_token_count,
    parse_bool_signal,
    parse_float_signal,
    parse_int_signal,
)

async def finalize_producer_task(producer_task: asyncio.Task[Any]) -> None:
    """Cancel and await one chat producer task.

    Example:
        If the SSE stream closes early, the helper still cancels the producer
        task and waits briefly for its cleanup to finish.
    """
    if not producer_task.done():
        logger.debug(format_log_event("chat.request", "producer_cancelling"))
        producer_task.cancel()
    try:
        await asyncio.wait_for(producer_task, timeout=1)
    except asyncio.CancelledError:
        pass
    logger.debug(format_log_event("chat.request", "producer_cancelled"))


@login_required
def ai_main(request: HttpRequest):
    """Render the main AI chat page."""
    return render(request, "ai/ai_main.html", {"ai_model_name": get_model_name()})


@login_required
async def run_chat(request: HttpRequest) -> DatastarResponse:
    """Handle an SSE chat stream from the AI agent."""
    signals = read_signals(request) or {}
    user_text = signals.get(
        "userInput", ""
    )  # 'user-input' becomes userInput (courtesy of Datastar)
    logger.debug(
        format_log_event("chat.request", "signals_received", keys=",".join(sorted(signals.keys())))
    )
    logger.debug(
        format_log_event("chat.input", "received", len=len(user_text), preview=build_log_preview(user_text))
    )
    session_input_tokens = parse_int_signal(signals, "sessionInputTokens")
    session_output_tokens = parse_int_signal(signals, "sessionOutputTokens")
    session_total_tokens = parse_int_signal(signals, "sessionTotalTokens")
    session_has_cost = parse_bool_signal(signals, "sessionHasCost")
    session_cost_usd = parse_float_signal(signals, "sessionCostUsd")
    chat_session_key = normalize_chat_session_key(signals.get("chatSessionKey"))
    user_id, username = await sync_to_async(
        lambda: (request.user.id, request.user.username), thread_sensitive=True
    )()
    can_view_answer_notes, can_view_raw_sql = await sync_to_async(
        lambda: (
            getattr(request.user, "has_perm", lambda _permission: False)("core.view_answer_notes"),
            getattr(request.user, "has_perm", lambda _permission: False)("core.view_raw_sql"),
        ),
        thread_sensitive=True,
    )()
    if request.session.session_key is None:
        await sync_to_async(request.session.save, thread_sensitive=True)()
    thread_id = build_thread_id(
        user_id=user_id,
        session_key=str(request.session.session_key),
        client_key=chat_session_key,
    )
    conversation_code = build_public_conversation_code(thread_id)
    logger.debug(
        format_log_event(
            "chat.request",
            "ready",
            thread=build_short_log_id(thread_id),
            user_len=len(user_text),
        )
    )

    async def chat_updates() -> AsyncGenerator[DatastarEvent, Any]:
        """Yield Datastar SSE patches for one chat request lifecycle."""
        nonlocal session_input_tokens
        nonlocal session_output_tokens
        nonlocal session_total_tokens
        nonlocal session_has_cost
        nonlocal session_cost_usd

        robot_id = f"{ROBOT_ID_PREFIX}{uuid.uuid4().hex}"
        queue: asyncio.Queue[Any | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stream_timeout_seconds = 6 * 60 # timeout for full circle starting from user input to final response
        if not user_text:
            return

        producer_task = asyncio.create_task(
            run_chat_session(
                user_text=user_text,
                robot_id=robot_id,
                thread_id=thread_id,
                queue=queue,
                stream_timeout_seconds=stream_timeout_seconds,
            )
        )

        yield SSE.patch_signals(
            {
                "isWaitingResponse": True,
                "thinkingSeconds": 0,
                "UserFacingProgressMessage": "is thinking...",
                "chatSessionKey": chat_session_key,
            }
        )
        yield append_user_message(user_text)
        yield append_robot_container(robot_id)
        robot_text = ""
        stream_interrupted = False
        try:
            start_time = loop.time()
            last_timer_seconds = -1
            while True:
                elapsed_seconds = int(loop.time() - start_time)
                if elapsed_seconds != last_timer_seconds:
                    last_timer_seconds = elapsed_seconds
                    yield SSE.patch_signals({"thinkingSeconds": elapsed_seconds})
                if loop.time() - start_time > stream_timeout_seconds:
                    producer_task.cancel()
                    timeout_message = build_timeout_message()
                    robot_text += timeout_message
                    yield append_robot_text(
                        robot_id,
                        build_user_facing_message(robot_text, conversation_code),
                    )
                    break
                if producer_task.done() and queue.empty():
                    try:
                        exc = producer_task.exception()
                    except asyncio.CancelledError:
                        exc = None
                    if exc:
                        error_message = build_error_message(exc)
                        robot_text += error_message
                        yield append_robot_text(
                            robot_id,
                            build_user_facing_message(robot_text, conversation_code),
                        )
                    break
                try:
                    queue_event = await asyncio.wait_for(queue.get(), timeout=0.2)
                except TimeoutError:
                    continue
                if queue_event is None:
                    break
                if isinstance(queue_event, dict):
                    if queue_event.get("kind") == "usage":
                        usage_metadata = queue_event.get("usage_metadata")
                        if isinstance(usage_metadata, dict):
                            input_tokens = int(usage_metadata.get("input_tokens") or 0)
                            output_tokens = int(
                                usage_metadata.get("output_tokens") or 0
                            )
                            total_tokens = int(usage_metadata.get("total_tokens") or 0)
                            cost_usd_value = queue_event.get("cost_usd")
                            has_cost = cost_usd_value is not None
                            cost_usd = float(cost_usd_value or 0)
                            session_input_tokens += input_tokens
                            session_output_tokens += output_tokens
                            session_total_tokens += total_tokens
                            if has_cost:
                                session_has_cost = True
                                session_cost_usd += cost_usd
                            yield SSE.patch_signals(
                                {
                                    "LatestUsageMetadata": usage_metadata,
                                    "LatestUsageModel": str(
                                        queue_event.get("pricing_model") or queue_event.get("model") or ""
                                    ),
                                    "sessionInputTokens": session_input_tokens,
                                    "sessionInputTokensText": format_token_count(
                                        session_input_tokens
                                    ),
                                    "sessionOutputTokens": session_output_tokens,
                                    "sessionOutputTokensText": format_token_count(
                                        session_output_tokens
                                    ),
                                    "sessionTotalTokens": session_total_tokens,
                                    "sessionHasCost": session_has_cost,
                                    "sessionCostUsd": session_cost_usd,
                                    "sessionCostUsdText": format_cost_usd(
                                        session_cost_usd
                                    ),
                                    "sessionTokenBadgeText": format_compact_tokens(
                                        session_total_tokens
                                    ),
                                }
                            )
                        continue
                    if queue_event.get("kind") == "theme":
                        selected_theme = str(queue_event.get("theme") or "").strip()
                        if selected_theme:
                            logger.debug(
                                format_log_event(
                                    "tool.ui",
                                    "theme_switch",
                                    thread=build_short_log_id(thread_id),
                                    theme=selected_theme,
                                )
                            )
                            yield SSE.patch_signals(
                                {"ThemeSwitchCommand": selected_theme}
                            )
                        continue
                    if queue_event.get("kind") == "progress":
                        progress_text = str(queue_event.get("text") or "").strip()
                        if progress_text:
                            yield SSE.patch_signals(
                                {"UserFacingProgressMessage": progress_text}
                            )
                        continue
                    if queue_event.get("kind") == "token":
                        visible_prefix = robot_text
                        chunk = str(queue_event.get("text") or "")
                        display_chunk = str(queue_event.get("display_text") or chunk)
                        robot_text += chunk
                        yield append_robot_text(
                            robot_id, visible_prefix + display_chunk
                        )
                        continue
                    if queue_event.get("kind") == "blocks":
                        blocks = queue_event.get("blocks")
                        if isinstance(blocks, list):
                            robot_text = build_blocks_visible_text(blocks)
                            yield append_robot_blocks(
                                robot_id,
                                blocks,
                                can_view_answer_notes=can_view_answer_notes,
                                can_view_raw_sql=can_view_raw_sql,
                            )
                        continue
        except (asyncio.CancelledError, GeneratorExit):
            stream_interrupted = True
            logger.debug(
                format_log_event(
                    "chat.request",
                    "stream_interrupted",
                    thread=build_short_log_id(thread_id),
                    robot=robot_id.removeprefix(ROBOT_ID_PREFIX)[:8],
                )
            )
            raise
        finally:
            if not stream_interrupted and not producer_task.done() and not robot_text:
                logger.debug(
                    format_log_event(
                        "chat.request",
                        "stream_interrupted",
                        thread=build_short_log_id(thread_id),
                        robot=robot_id.removeprefix(ROBOT_ID_PREFIX)[:8],
                    )
                )
            cleanup_task = asyncio.create_task(finalize_producer_task(producer_task))
            try:
                await asyncio.shield(cleanup_task)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                logger.exception(
                    format_log_event(
                        "chat.request",
                        "cleanup_failed",
                        thread=build_short_log_id(thread_id),
                    )
                )
            logger.debug(
                format_log_event(
                    "chat.request",
                    "response_ready",
                    thread=build_short_log_id(thread_id),
                    content_len=len(robot_text),
                    preview=build_log_preview(robot_text),
                )
            )
            if not stream_interrupted and robot_text:
                append_chat_log_turn(username, thread_id, user_text, robot_text)
            if not stream_interrupted:
                yield SSE.patch_signals(
                    {
                        "isWaitingResponse": False,
                        "thinkingSeconds": 0,
                        "UserFacingProgressMessage": "is thinking...",
                        "ThemeSwitchCommand": "",
                    }
                )

    return DatastarResponse(chat_updates())
