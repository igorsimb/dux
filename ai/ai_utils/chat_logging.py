"""File-based helpers for chat transcript logging."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from django.conf import settings
from django.utils import timezone
from loguru import logger

from .logging_config import build_log_preview, build_public_conversation_code, format_log_event

CHAT_LOGS_DIRNAME: Final = "chatlogs"
THREAD_SUFFIX_LENGTH: Final = 5
USERNAME_SAFE_CHARS_RE: Final = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_chat_log_username(username: str) -> str:
    """Return a filesystem-safe username while preserving `.`, `_`, and `-`."""
    sanitized = USERNAME_SAFE_CHARS_RE.sub("_", (username or "").strip())
    return sanitized or "user"


def build_chat_log_path(username: str, thread_id: str, *, now=None) -> Path:
    """Build the transcript file path for one user/chat/day."""
    local_now = timezone.localtime(now or timezone.now())
    safe_username = sanitize_chat_log_username(username)
    thread_suffix = (thread_id or "")[-THREAD_SUFFIX_LENGTH:] or "chat"
    filename = f"{local_now:%d-%m-%Y}-{thread_suffix}.txt"
    return Path(settings.BASE_DIR) / CHAT_LOGS_DIRNAME / safe_username / filename


def format_chat_log_turn(user_text: str, ai_text: str, *, now=None) -> str:
    """Format one transcript turn as a readable plain-text block."""
    local_now = timezone.localtime(now or timezone.now())
    return f"[{local_now:%Y-%m-%d %H:%M:%S}]\nUser: {user_text}\nAI: {ai_text}\n\n"


def format_chat_log_header(thread_id: str) -> str:
    """Format the stable transcript header for one conversation."""
    return f"Conversation: {build_public_conversation_code(thread_id)}\n\n"


def append_chat_log_turn(
    username: str, thread_id: str, user_text: str, ai_text: str, *, now=None
) -> Path | None:
    """Append one completed turn to the current chat log file.

    Returns the written path on success. If file I/O fails, logs a warning and returns ``None``.
    """
    path = build_chat_log_path(username, thread_id, now=now)
    header_block = format_chat_log_header(thread_id)
    turn_block = format_chat_log_turn(user_text, ai_text, now=now)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log_file:
            if path.stat().st_size == 0:
                log_file.write(header_block)
            log_file.write(turn_block)
    except OSError as exc:
        logger.warning(
            format_log_event(
                "chat.transcript",
                "write_failed",
                path=path,
                error=build_log_preview(exc),
            )
        )
        return None
    return path
