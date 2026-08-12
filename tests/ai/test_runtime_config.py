from __future__ import annotations

from ai.ai_utils import runtime_config


def test_get_model_name_prefers_openai_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    monkeypatch.setenv("MODEL_NAME", "gpt-legacy")

    assert runtime_config.get_model_name() == "gpt-5.4"


def test_get_model_name_falls_back_to_legacy_model_name(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-5.4")

    assert runtime_config.get_model_name() == "gpt-5.4"
