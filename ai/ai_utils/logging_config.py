"""Centralized Loguru configuration for persistent application logs."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings
from loguru import logger

from .chat_session import THREAD_ID_PREFIX

LOG_DIR = "logs"
LOG_LEVEL = "DEBUG"
LOG_ROTATION = "100 MB"
LOG_RETENTION = "14 days"
LOG_JSON = False  # False = plain text, True = JSON
LOG_AREA_WIDTH = 16
LOG_EVENT_WIDTH = 18
LOG_DETAIL_INDENT = " " * 11

CONVERSATION_CODE_LENGTH = 12
LOG_FORMAT = "{time:HH:mm:ss.SSS} {level:<5} {message}"


@dataclass(frozen=True)
class LogPaths:
    debug: Path
    error: Path


def build_log_paths(log_dir: str | Path = LOG_DIR) -> LogPaths:
    resolved_log_dir = _resolve_log_dir(log_dir)
    return LogPaths(
        debug=resolved_log_dir / "debug.log",
        error=resolved_log_dir / "error.log",
    )


def build_public_conversation_code(thread_id: str) -> str:
    """Return a short support-friendly code derived from a conversation id.

    Example:
        `chat-thread-1234567890abcdef...` becomes `1234567890ab`.

        Non-standard ids still produce a deterministic code:
        `legacy-conversation-id` becomes the first 12 chars of its SHA-256 digest.
    """
    digest = _extract_thread_digest(thread_id)
    if digest is None:
        digest = hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()
    return digest[:CONVERSATION_CODE_LENGTH]


def build_short_log_id(value: object, *, length: int = 8) -> str:
    """Return a compact stable identifier for live log scanning."""
    raw_value = str(value)
    digest = _extract_thread_digest(raw_value)
    if digest is not None:
        return digest[:length]
    return raw_value[:length]


def build_log_preview(value: object, *, limit: int = 80) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit - 1]}..."


def format_log_event(
    area: str,
    event: str,
    /,
    *,
    detail_lines: list[str] | tuple[str, ...] | None = None,
    **details: Any,
) -> str:
    """Build the app's readable structured log message body.

    The returned text is the message body passed to Loguru. The configured sink
    adds time and level before it, so:

        logger.debug(format_log_event("chat.request", "ready", thread="5f822109", user_len=11))

    is written as:

        11:10:56.065 DEBUG chat.request     ready              thread=5f822109 user_len=11

    Long secondary details can be emitted as indented lines:

        format_log_event(
            "chat.agent",
            "tools_ready",
            count=2,
            detail_lines=["tools: ask_user, validate_sql"],
        )

    returns:

        chat.agent       tools_ready        count=2
                   tools: ask_user, validate_sql
    """
    detail_text = " ".join(
        f"{_format_log_key(key)}={_format_log_value(value)}" for key, value in details.items() if value is not None
    )
    line = f"{area:<{LOG_AREA_WIDTH}} {event:<{LOG_EVENT_WIDTH}}"
    if detail_text:
        line = f"{line} {detail_text}"
    if not detail_lines:
        return line
    extra_lines = "\n".join(f"{LOG_DETAIL_INDENT}{line}" for line in detail_lines)
    return f"{line}\n{extra_lines}"


def configure_logging(*, log_dir: str | Path = LOG_DIR) -> LogPaths:
    paths = build_log_paths(log_dir)
    paths.debug.parent.mkdir(parents=True, exist_ok=True)
    paths.debug.touch(exist_ok=True)
    paths.error.touch(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        serialize=LOG_JSON,
    )
    logger.add(
        paths.debug,
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        serialize=LOG_JSON,
    )
    logger.add(
        paths.error,
        level="ERROR",
        format=LOG_FORMAT,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        serialize=LOG_JSON,
    )
    return paths


def _extract_thread_digest(thread_id: str) -> str | None:
    """Return the hex digest portion from a standard backend thread id.

    Example:
        `chat-thread-1234567890abcdef` returns `1234567890abcdef`.

        `chat-tab-1234` returns `None` because it is not a canonical backend thread id.
    """
    raw_thread_id = str(thread_id)
    if not raw_thread_id.startswith(THREAD_ID_PREFIX):
        return None
    digest = raw_thread_id[len(THREAD_ID_PREFIX) :]
    if len(digest) < CONVERSATION_CODE_LENGTH or not _is_hex_string(digest):
        return None
    return digest


def _is_hex_string(value: str) -> bool:
    return bool(value) and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _format_log_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if any(character.isspace() for character in text) or '"' in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\r", "\\r").replace("\n", "\\n")
        return f'"{escaped}"'
    return text


def _format_log_key(key: str) -> str:
    return key[:-1] if key.endswith("_") else key


def _resolve_log_dir(log_dir: str | Path) -> Path:
    candidate = Path(log_dir)
    if candidate.is_absolute():
        return candidate
    if settings.configured:
        return Path(settings.BASE_DIR) / candidate
    return Path.cwd() / candidate
