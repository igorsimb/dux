from __future__ import annotations

import pytest

from ai.ai_utils import runtime_config


def test_get_model_name_prefers_openai_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    monkeypatch.setenv("MODEL_NAME", "gpt-legacy")

    assert runtime_config.get_model_name() == "gpt-5.4"


def test_get_model_name_falls_back_to_legacy_model_name(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-5.4")

    assert runtime_config.get_model_name() == "gpt-5.4"


def test_get_enable_streaming_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_ENABLE_STREAMING", raising=False)

    assert runtime_config.get_enable_streaming() is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", " TRUE "])
def test_get_enable_streaming_accepts_enabled_values(monkeypatch, value) -> None:
    monkeypatch.setenv("OPENAI_ENABLE_STREAMING", value)

    assert runtime_config.get_enable_streaming() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", "   "])
def test_get_enable_streaming_rejects_non_enabled_values(monkeypatch, value) -> None:
    monkeypatch.setenv("OPENAI_ENABLE_STREAMING", value)

    assert runtime_config.get_enable_streaming() is False
