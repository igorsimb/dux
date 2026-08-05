"""Public exports for AI chat utility helpers used by views and runtime code."""

from typing import Any

from .chat_errors import (
    build_error_message,
    build_timeout_message,
    build_user_facing_message,
)
from .ui import (
    ROBOT_ID_PREFIX,
    append_robot_blocks,
    append_robot_container,
    append_robot_text,
    append_user_message,
    build_blocks_visible_text,
)


async def run_chat_session(*args: Any, **kwargs: Any) -> Any:
    from .chat_runtime import run_chat_session as _run_chat_session

    return await _run_chat_session(*args, **kwargs)


__all__ = [
    "ROBOT_ID_PREFIX",
    "append_robot_blocks",
    "append_robot_container",
    "append_robot_text",
    "append_user_message",
    "build_blocks_visible_text",
    "build_error_message",
    "build_timeout_message",
    "build_user_facing_message",
    "run_chat_session",
]
