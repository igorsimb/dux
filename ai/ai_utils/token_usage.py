"""Helpers for usage metadata, pricing, and compact usage signal formatting.

Pricing is intentionally kept local and tiny for the models this app actually
uses. Sync values manually from OpenAI pricing docs when needed:

- https://openai.com/api/pricing/
- https://developers.openai.com/api/docs/pricing
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class OpenAIModelPricing:
    """Per-1M token prices for one OpenAI model.

    Example:
        >>> OpenAIModelPricing(input_per_million=Decimal("2.50"), cached_input_per_million=Decimal("0.25"), output_per_million=Decimal("15.00"))
        OpenAIModelPricing(input_per_million=Decimal('2.50'), cached_input_per_million=Decimal('0.25'), output_per_million=Decimal('15.00'))
    """

    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


OPENAI_PRICING_BY_MODEL: dict[str, OpenAIModelPricing] = {
    # The model for flagship capability.
    "gpt-5.6-sol": OpenAIModelPricing(
        input_per_million=Decimal("5.00"),
        cached_input_per_million=Decimal("0.50"),
        output_per_million=Decimal("30.00"),
    ),
    # Strong performance at a lower price.
    "gpt-5.6-terra": OpenAIModelPricing(
        input_per_million=Decimal("2.50"),
        cached_input_per_million=Decimal("0.25"),
        output_per_million=Decimal("15.00"),
    ),
    # Efficient, high-volume workloads.
    "gpt-5.6-luna": OpenAIModelPricing(
        input_per_million=Decimal("1.00"),
        cached_input_per_million=Decimal("0.10"),
        output_per_million=Decimal("6.00"),
    ),
    "gpt-5.5": OpenAIModelPricing(
        input_per_million=Decimal("5.00"),
        cached_input_per_million=Decimal("0.50"),
        output_per_million=Decimal("45.00"),
    ),
    "gpt-5.4": OpenAIModelPricing(
        input_per_million=Decimal("2.50"),
        cached_input_per_million=Decimal("0.25"),
        output_per_million=Decimal("15.00"),
    ),
    "gpt-5.2": OpenAIModelPricing(
        input_per_million=Decimal("1.75"),
        cached_input_per_million=Decimal("0.175"),
        output_per_million=Decimal("14.00"),
    ),
}


def parse_int_signal(signals: dict[str, Any], key: str) -> int:
    """Return an integer signal value, defaulting to zero.

    Example:
        >>> parse_int_signal({"sessionTotalTokens": "1200"}, "sessionTotalTokens")
        1200
    """
    value = signals.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_float_signal(signals: dict[str, Any], key: str) -> float:
    """Return a float signal value, defaulting to zero.

    Example:
        >>> parse_float_signal({"sessionCostUsd": "0.0125"}, "sessionCostUsd")
        0.0125
    """
    value = signals.get(key)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_bool_signal(signals: dict[str, Any], key: str) -> bool:
    """Return a boolean signal value with string-aware parsing.

    Examples:
        >>> parse_bool_signal({"sessionHasCost": "true"}, "sessionHasCost")
        True
        >>> parse_bool_signal({"sessionHasCost": "0"}, "sessionHasCost")
        False
    """
    value = signals.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def format_compact_tokens(token_count: int) -> str:
    """Return a compact token label for the small header badge.

    Example:
        >>> format_compact_tokens(1800)
        '1.8k ток'
    """
    if token_count >= 1_000_000:
        value = token_count / 1_000_000
        return f"{value:.0f}M ток" if token_count >= 10_000_000 else f"{value:.1f}M ток"
    if token_count >= 1_000:
        value = token_count / 1_000
        return f"{value:.0f}k ток" if token_count >= 10_000 else f"{value:.1f}k ток"
    return f"{token_count} ток"


def format_token_count(token_count: int) -> str:
    """Return a full token count with grouped thousands.

    Example:
        >>> format_token_count(1234567)
        '1 234 567'
    """
    return f"{token_count:,}".replace(",", " ")


def format_cost_usd(cost_usd: float) -> str:
    """Return a readable USD number string for the usage breakdown.

    Example:
        >>> format_cost_usd(0.0042)
        '0.0042'
    """
    if cost_usd >= 1:
        return f"{cost_usd:.2f}"
    if cost_usd >= 0.01:
        return f"{cost_usd:.4f}"
    return f"{cost_usd:.6f}".rstrip("0").rstrip(".")


def calculate_openai_usage_cost(
    model_name: str, usage_metadata: dict[str, Any]
) -> float | None:
    """Return a USD estimate for one usage payload.

    Example:
        >>> calculate_openai_usage_cost(
        ...     "gpt-5.4",
        ...     {"input_tokens": 1000, "output_tokens": 500, "input_token_details": {"cache_read": 100}},
        ... )
        0.009

    If detailed cached-input counts are absent, all input tokens are priced at
    the standard input rate.
    """
    pricing = OPENAI_PRICING_BY_MODEL.get(model_name)
    if pricing is None:
        return None

    input_tokens = int(usage_metadata.get("input_tokens") or 0)
    output_tokens = int(usage_metadata.get("output_tokens") or 0)
    input_token_details = usage_metadata.get("input_token_details")
    cache_read_tokens = 0
    if isinstance(input_token_details, dict):
        cache_read_tokens = int(input_token_details.get("cache_read") or 0)

    cache_read_tokens = max(0, min(cache_read_tokens, input_tokens))
    standard_input_tokens = input_tokens - cache_read_tokens

    input_cost = _cost_for_tokens(standard_input_tokens, pricing.input_per_million)
    cached_input_cost = _cost_for_tokens(
        cache_read_tokens, pricing.cached_input_per_million
    )
    output_cost = _cost_for_tokens(output_tokens, pricing.output_per_million)
    total_cost = input_cost + cached_input_cost + output_cost
    return float(total_cost)


def _cost_for_tokens(token_count: int, price_per_million: Decimal) -> Decimal:
    """Return the USD cost for ``token_count`` at ``price_per_million``.

    Example:
        >>> _cost_for_tokens(1000, Decimal("2.50"))
        Decimal('0.00250')
    """
    if token_count <= 0:
        return Decimal("0")
    return (Decimal(token_count) / Decimal(1_000_000)) * price_per_million
