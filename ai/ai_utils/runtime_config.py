"""Runtime configuration helpers for reading OpenAI-related environment variables."""

from __future__ import annotations

import os


def get_model_name() -> str:
    """Return the OpenAI model name for chat requests.

    Example:
        If `OPENAI_MODEL` is unset but legacy `MODEL_NAME` is present, this
        helper still returns the configured model name.
    """
    return os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME") or "gpt-5.4"


def get_api_key() -> str | None:
    """Return the OpenAI API key from environment variables."""
    return os.getenv("OPENAI_API_KEY")


def get_openai_proxy() -> str | None:
    """Return optional OpenAI proxy URL."""
    return os.getenv("OPENAI_PROXY")
