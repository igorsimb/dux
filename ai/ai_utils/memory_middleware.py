"""Token-budget middleware for trimming chat history before model calls."""

from __future__ import annotations

from typing import Any, Sequence, cast

from langchain.agents.middleware import before_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from loguru import logger

from .agent_state import ChatAgentState
from .logging_config import format_log_event

MAX_HISTORY_TOKENS = 24000
TOKEN_COUNTER_NAME = "count_tokens_approximately"


def _approximate_tokens(messages: Sequence[BaseMessage]) -> int:
    """Estimate token usage for a message sequence.

    Example:
        >>> _approximate_tokens([HumanMessage(content="hello")]) > 0
        True
    """

    return int(count_tokens_approximately(messages)) if messages else 0


def _messages_match(
    messages: Sequence[BaseMessage], trimmed_messages: Sequence[BaseMessage]
) -> bool:
    """Return True when trimming leaves the message list unchanged.

    Example:
        >>> _messages_match([HumanMessage(content="hi")], [HumanMessage(content="hi")])
        True
    """

    return len(messages) == len(trimmed_messages) and all(
        left == right for left, right in zip(messages, trimmed_messages)
    )


def _log_trim_decision(
    *,
    was_trimmed: bool,
    message_count_before: int,
    message_count_after: int,
    approx_tokens_before: int,
    approx_tokens_after: int,
) -> None:
    event = "trim_applied" if was_trimmed else "trim_skipped"
    logger.debug(
        format_log_event(
            "chat.memory",
            event,
            before=message_count_before,
            after=message_count_after,
            tokens_before=approx_tokens_before,
            tokens_after=approx_tokens_after,
            counter=TOKEN_COUNTER_NAME,
        )
    )


def _most_recent_actionable_turn(messages: Sequence[object]) -> list[Any]:
    """Keep the last actionable turn when strict trimming would drop everything.

    This fail-open fallback prevents the middleware from clearing all history when
    the latest oversized turn alone exceeds the token budget.

    Examples:
        >>> _most_recent_actionable_turn([HumanMessage(content="hello")])[0].content
        'hello'
        >>> len(_most_recent_actionable_turn([
        ...     HumanMessage(content="q"),
        ...     AIMessage(content="", tool_calls=[{"id": "c1", "name": "lookup", "args": {"q": "x"}}]),
        ...     ToolMessage(content="result", tool_call_id="c1"),
        ... ]))
        3
    """

    if not messages:
        return []

    last_message = cast(BaseMessage, messages[-1])
    if isinstance(last_message, HumanMessage):
        return [last_message]
    if isinstance(last_message, ToolMessage):
        for index in range(len(messages) - 1, -1, -1):
            message = cast(BaseMessage, messages[index])
            if isinstance(message, HumanMessage):
                return [cast(BaseMessage, item) for item in messages[index:]]
        return [last_message]
    if isinstance(last_message, AIMessage):
        return [last_message]
    return [last_message]


def trim_chat_history(
    state: ChatAgentState,
    runtime: Runtime[Any],
) -> dict[str, Any] | None:
    """Trim message history to a token budget before each model call.

    The middleware only updates `messages`. Other state, including
    `validated_queries`, stays untouched.

    Examples:
        >>> trim_chat_history({"messages": []}, Runtime()) is None
        True
        >>> result = trim_chat_history(
        ...     {"messages": [HumanMessage(content="hello")]},
        ...     Runtime(),
        ... )
        >>> result is None
        True
    """

    messages = cast(list[BaseMessage], list(state.get("messages") or []))
    message_count_before = len(messages)
    approx_tokens_before = _approximate_tokens(messages)

    if not messages:
        _log_trim_decision(
            was_trimmed=False,
            message_count_before=0,
            message_count_after=0,
            approx_tokens_before=0,
            approx_tokens_after=0,
        )
        return None

    trimmed_messages = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_HISTORY_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
        allow_partial=False,
    )
    if not trimmed_messages:
        trimmed_messages = _most_recent_actionable_turn(messages)
    approx_tokens_after = _approximate_tokens(trimmed_messages)
    was_trimmed = not _messages_match(messages, trimmed_messages)

    _log_trim_decision(
        was_trimmed=was_trimmed,
        message_count_before=message_count_before,
        message_count_after=len(trimmed_messages),
        approx_tokens_before=approx_tokens_before,
        approx_tokens_after=approx_tokens_after,
    )
    if not was_trimmed:
        return None

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed_messages]}


trim_chat_history_middleware = before_model(state_schema=ChatAgentState)(
    trim_chat_history
)
