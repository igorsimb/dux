from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from loguru import logger

from ai.ai_utils import logging_config


@pytest.fixture(autouse=True)
def reset_loguru_sinks() -> None:
    logger.remove()
    yield
    logger.remove()
    logger.add(sys.stderr)


def test_build_log_paths_returns_debug_and_error_log_paths(tmp_path: Path) -> None:
    paths = logging_config.build_log_paths(tmp_path)

    assert paths.debug == tmp_path / "debug.log"
    assert paths.error == tmp_path / "error.log"


def test_build_log_paths_uses_base_dir_logs_by_default(
    settings, tmp_path: Path
) -> None:
    settings.BASE_DIR = tmp_path

    paths = logging_config.build_log_paths()

    assert paths.debug == tmp_path / "logs" / "debug.log"
    assert paths.error == tmp_path / "logs" / "error.log"


def test_build_public_conversation_code_uses_thread_digest_prefix() -> None:
    thread_id = (
        "chat-thread-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    )

    assert logging_config.build_public_conversation_code(thread_id) == "1234567890ab"


def test_build_public_conversation_code_hashes_nonstandard_thread_ids() -> None:
    thread_id = "not-a-standard-thread-id"
    expected = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:12]

    assert logging_config.build_public_conversation_code(thread_id) == expected


def test_build_short_log_id_uses_thread_digest_prefix() -> None:
    thread_id = (
        "chat-thread-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    )

    assert logging_config.build_short_log_id(thread_id) == "12345678"


def test_configure_logging_creates_log_directory_and_files(tmp_path: Path) -> None:
    logging_config.configure_logging(log_dir=tmp_path)

    logger.debug("phase2-debug-message")
    logger.error("phase2-error-message")
    logger.complete()

    debug_log = tmp_path / "debug.log"
    error_log = tmp_path / "error.log"

    assert debug_log.exists()
    assert error_log.exists()
    assert "phase2-debug-message" in debug_log.read_text(encoding="utf-8")
    assert "phase2-error-message" in debug_log.read_text(encoding="utf-8")
    assert "phase2-error-message" in error_log.read_text(encoding="utf-8")


def test_configure_logging_keeps_debug_only_out_of_error_log(tmp_path: Path) -> None:
    logging_config.configure_logging(log_dir=tmp_path)

    logger.debug("debug-only-message")
    logger.complete()

    error_log = tmp_path / "error.log"

    assert error_log.exists()
    assert "debug-only-message" not in error_log.read_text(encoding="utf-8")


def test_configure_logging_is_effectively_idempotent_for_file_sinks(
    tmp_path: Path,
) -> None:
    logging_config.configure_logging(log_dir=tmp_path)
    logging_config.configure_logging(log_dir=tmp_path)

    logger.info("single-copy-message")
    logger.complete()

    debug_log = tmp_path / "debug.log"
    content = debug_log.read_text(encoding="utf-8")

    assert content.count("single-copy-message") == 1
