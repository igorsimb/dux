from __future__ import annotations

import pytest

from ai.ai_utils.token_usage import calculate_openai_usage_cost


@pytest.mark.parametrize(
    ("model_name", "expected_cost_usd"),
    [
        ("gpt-5.6-sol", 35.0),
        ("gpt-5.6-terra", 17.5),
        ("gpt-5.6-luna", 7.0),
    ],
)
def test_calculate_openai_usage_cost_supports_gpt_5_6_models(
    model_name: str,
    expected_cost_usd: float,
) -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

    assert calculate_openai_usage_cost(model_name, usage) == expected_cost_usd


def test_calculate_openai_usage_cost_prices_standard_input_and_output() -> None:
    cost_usd = calculate_openai_usage_cost(
        "gpt-5.4",
        {"input_tokens": 1_000, "output_tokens": 500, "total_tokens": 1_500},
    )

    assert cost_usd == 0.01


def test_calculate_openai_usage_cost_uses_cached_input_pricing() -> None:
    cost_usd = calculate_openai_usage_cost(
        "gpt-5.4",
        {
            "input_tokens": 1_000,
            "output_tokens": 500,
            "total_tokens": 1_500,
            "input_token_details": {"cache_read": 100},
        },
    )

    assert cost_usd == 0.009775


def test_calculate_openai_usage_cost_returns_none_for_unknown_model() -> None:
    assert calculate_openai_usage_cost("unknown-model", {"input_tokens": 100}) is None


def test_calculate_openai_usage_cost_clamps_cache_read_to_input_tokens() -> None:
    cost_usd = calculate_openai_usage_cost(
        "gpt-5.4",
        {
            "input_tokens": 100,
            "output_tokens": 0,
            "input_token_details": {"cache_read": 999},
        },
    )

    assert cost_usd == 0.000025
