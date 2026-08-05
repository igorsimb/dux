from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ai.ai_utils.chat_logging import (
    append_chat_log_turn,
    build_chat_log_path,
    format_chat_log_turn,
    sanitize_chat_log_username,
)
from ai.ai_utils.logging_config import build_public_conversation_code


def test_sanitize_chat_log_username_preserves_dot_dash_and_underscore() -> None:
    assert sanitize_chat_log_username("i.dolgikh") == "i.dolgikh"
    assert sanitize_chat_log_username("i_dolgikh-test") == "i_dolgikh-test"
    assert sanitize_chat_log_username("sample/user:@test") == "sample_user_test"


def test_build_chat_log_path_uses_local_date_and_thread_suffix(
    settings, tmp_path: Path
) -> None:
    settings.BASE_DIR = tmp_path
    now = datetime(2026, 3, 24, 14, 8, 11, tzinfo=ZoneInfo("Europe/Moscow"))

    path = build_chat_log_path("i.dolgikh", "chat-thread-123456aa7a", now=now)

    assert path == tmp_path / "chatlogs" / "i.dolgikh" / "24-03-2026-6aa7a.txt"


def test_format_chat_log_turn_builds_readable_plain_text_block(settings) -> None:
    now = datetime(2026, 3, 24, 14, 8, 11, tzinfo=ZoneInfo("Europe/Moscow"))

    assert format_chat_log_turn("Привет", "Здравствуйте", now=now) == (
        "[2026-03-24 14:08:11]\nUser: Привет\nAI: Здравствуйте\n\n"
    )


def test_append_chat_log_turn_creates_parent_and_appends_turns(
    settings, tmp_path: Path
) -> None:
    settings.BASE_DIR = tmp_path
    now = datetime(2026, 3, 24, 14, 8, 11, tzinfo=ZoneInfo("Europe/Moscow"))

    path = append_chat_log_turn(
        "i.dolgikh", "chat-thread-123456aa7a", "Привет", "Здравствуйте", now=now
    )
    second_path = append_chat_log_turn(
        "i.dolgikh",
        "chat-thread-123456aa7a",
        "Что умеешь?",
        "Помогаю с данными.",
        now=now,
    )

    assert path == tmp_path / "chatlogs" / "i.dolgikh" / "24-03-2026-6aa7a.txt"
    assert second_path == path
    assert path.read_text(encoding="utf-8") == (
        f"Conversation: {build_public_conversation_code('chat-thread-123456aa7a')}\n\n"
        "[2026-03-24 14:08:11]\n"
        "User: Привет\n"
        "AI: Здравствуйте\n\n"
        "[2026-03-24 14:08:11]\n"
        "User: Что умеешь?\n"
        "AI: Помогаю с данными.\n\n"
    )


def test_append_chat_log_turn_keeps_existing_filename_convention(
    settings, tmp_path: Path
) -> None:
    settings.BASE_DIR = tmp_path
    now = datetime(2026, 3, 24, 14, 8, 11, tzinfo=ZoneInfo("Europe/Moscow"))

    path = append_chat_log_turn(
        "i.dolgikh", "chat-thread-123456aa7a", "Привет", "Здравствуйте", now=now
    )

    assert path.name == "24-03-2026-6aa7a.txt"


def test_append_chat_log_turn_returns_none_and_logs_warning_on_write_failure(
    monkeypatch, settings, tmp_path: Path
) -> None:
    settings.BASE_DIR = tmp_path
    now = datetime(2026, 3, 24, 14, 8, 11, tzinfo=ZoneInfo("Europe/Moscow"))
    warning_calls: list[tuple[str, tuple[object, ...]]] = []

    def fake_open(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", fake_open)
    monkeypatch.setattr(
        "ai.ai_utils.chat_logging.logger.warning",
        lambda message, *args: warning_calls.append((str(message), args)),
    )

    result = append_chat_log_turn(
        "i.dolgikh", "chat-thread-123456aa7a", "Привет", "Здравствуйте", now=now
    )

    assert result is None
    assert len(warning_calls) == 1
    message, args = warning_calls[0]
    assert args == ()
    assert "chat.transcript" in message
    assert "write_failed" in message
    assert str(tmp_path / "chatlogs" / "i.dolgikh" / "24-03-2026-6aa7a.txt") in message
    assert "disk full" in message
