"""Runtime configuration helpers for reading OpenAI-related environment variables."""

from __future__ import annotations

import os


def get_model_name() -> str:
    """Return the OpenAI model name for chat requests."""
    return os.getenv("OPENAI_MODEL") or "gpt-5.6"


def get_api_key() -> str | None:
    """Return the OpenAI API key from environment variables."""
    return os.getenv("OPENAI_API_KEY")


def get_openai_proxy() -> str | None:
    """Return optional OpenAI proxy URL."""
    return os.getenv("OPENAI_PROXY")
