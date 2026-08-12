from __future__ import annotations

from ai.ai_utils import runtime_config


def test_get_model_name_prefers_openai_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")

    assert runtime_config.get_model_name() == "gpt-5.4"


def test_get_model_name_uses_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert runtime_config.get_model_name() == "gpt-5.6"
