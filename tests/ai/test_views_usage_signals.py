from __future__ import annotations

import asyncio
from typing import Any, cast

from ai import views
from ai.ai_utils import token_usage


class _FakeSession:
    def __init__(self, session_key: str) -> None:
        self.session_key = session_key


def _usage_test_request(session_key: str) -> object:
    return type(
        "Request",
        (),
        {
            "user": type("User", (), {"id": 17, "username": "usage-tester"})(),
            "session": _FakeSession(session_key),
        },
    )()


def _usage_event_payload(events: list[dict[str, object]]) -> dict[str, object]:
    usage_event = next(
        event
        for event in events
        if event["kind"] == "signals"
        and "LatestUsageMetadata" in cast(dict[str, object], event["payload"])
    )
    return cast(dict[str, object], usage_event["payload"])


def test_format_compact_tokens_formats_small_and_large_values() -> None:
    assert token_usage.format_compact_tokens(999) == "999 ток"
    assert token_usage.format_compact_tokens(1_800) == "1.8k ток"
    assert token_usage.format_compact_tokens(12_000) == "12k ток"


def test_format_usage_breakdown_values() -> None:
    assert token_usage.format_token_count(1_234_567) == "1 234 567"
    assert token_usage.format_cost_usd(1.2345) == "1.23"
    assert token_usage.format_cost_usd(0.1234) == "0.1234"
    assert token_usage.format_cost_usd(0.0042) == "0.0042"


def test_parse_signal_helpers_fallback_cleanly() -> None:
    signals = {"sessionTotalTokens": "1200", "sessionCostUsd": "0.0125"}

    assert token_usage.parse_int_signal(signals, "sessionTotalTokens") == 1200
    assert token_usage.parse_int_signal(signals, "missing") == 0
    assert token_usage.parse_float_signal(signals, "sessionCostUsd") == 0.0125
    assert (
        token_usage.parse_float_signal({"sessionCostUsd": "bad"}, "sessionCostUsd")
        == 0.0
    )


def test_parse_bool_signal_handles_boolean_like_strings() -> None:
    assert (
        token_usage.parse_bool_signal({"sessionHasCost": "true"}, "sessionHasCost")
        is True
    )
    assert (
        token_usage.parse_bool_signal({"sessionHasCost": "1"}, "sessionHasCost") is True
    )
    assert (
        token_usage.parse_bool_signal({"sessionHasCost": "false"}, "sessionHasCost")
        is False
    )
    assert (
        token_usage.parse_bool_signal({"sessionHasCost": "0"}, "sessionHasCost")
        is False
    )
    assert (
        token_usage.parse_bool_signal({"sessionHasCost": ""}, "sessionHasCost") is False
    )


def test_run_chat_keeps_session_token_counters_as_cumulative_frontend_metrics(
    monkeypatch,
) -> None:
    request = _usage_test_request("django-session-usage")

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "здарова",
            "chatSessionKey": "chat-tab-usage",
            "sessionInputTokens": "100",
            "sessionOutputTokens": "40",
            "sessionTotalTokens": "140",
            "sessionCostUsd": "0.25",
            "sessionHasCost": True,
        },
    )
    monkeypatch.setattr(views, "DatastarResponse", lambda generator: generator)
    monkeypatch.setattr(
        views, "append_user_message", lambda text: {"kind": "user", "text": text}
    )
    monkeypatch.setattr(
        views,
        "append_robot_container",
        lambda robot_id: {"kind": "robot-container", "robot_id": robot_id},
    )
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def fake_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        await queue.put(
            {
                "kind": "usage",
                "model": "gpt-test",
                "pricing_model": "gpt-env",
                "cost_usd": 0.5,
                "usage_metadata": {
                    "input_tokens": 25,
                    "output_tokens": 15,
                    "total_tokens": 40,
                },
            }
        )
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> list[dict[str, object]]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        events: list[dict[str, object]] = []
        async for event in stream:
            events.append(event)
        return events

    events = asyncio.run(exercise())
    payload = _usage_event_payload(events)

    assert payload == {
        "LatestUsageMetadata": {
            "input_tokens": 25,
            "output_tokens": 15,
            "total_tokens": 40,
        },
        "LatestUsageModel": "gpt-env",
        "sessionInputTokens": 125,
        "sessionInputTokensText": "125",
        "sessionOutputTokens": 55,
        "sessionOutputTokensText": "55",
        "sessionTotalTokens": 180,
        "sessionHasCost": True,
        "sessionCostUsd": 0.75,
        "sessionCostUsdText": "0.7500",
        "sessionTokenBadgeText": "180 ток",
    }


def test_run_chat_updates_usage_signals_for_theme_change_turn(monkeypatch) -> None:
    request = _usage_test_request("django-session-theme-usage")

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "смени тему на nord",
            "chatSessionKey": "chat-tab-theme-usage",
            "sessionInputTokens": "10",
            "sessionOutputTokens": "5",
            "sessionTotalTokens": "15",
            "sessionCostUsd": "0.0100",
            "sessionHasCost": True,
        },
    )
    monkeypatch.setattr(views, "DatastarResponse", lambda generator: generator)
    monkeypatch.setattr(
        views, "append_user_message", lambda text: {"kind": "user", "text": text}
    )
    monkeypatch.setattr(
        views,
        "append_robot_container",
        lambda robot_id: {"kind": "robot-container", "robot_id": robot_id},
    )
    monkeypatch.setattr(
        views,
        "append_robot_text",
        lambda robot_id, text: {
            "kind": "robot-text",
            "robot_id": robot_id,
            "text": text,
        },
    )
    monkeypatch.setattr(views, "append_chat_log_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def fake_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        await queue.put({"kind": "theme", "theme": "nord"})
        await queue.put({"kind": "token", "text": "Готово."})
        await queue.put(
            {
                "kind": "usage",
                "model": "gpt-5.4",
                "pricing_model": "gpt-env",
                "cost_usd": 0.0025,
                "usage_metadata": {
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                },
            }
        )
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> list[dict[str, object]]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        return [event async for event in stream]

    events = asyncio.run(exercise())
    payload = _usage_event_payload(events)

    assert payload == {
        "LatestUsageMetadata": {
            "input_tokens": 20,
            "output_tokens": 8,
            "total_tokens": 28,
        },
        "LatestUsageModel": "gpt-env",
        "sessionInputTokens": 30,
        "sessionInputTokensText": "30",
        "sessionOutputTokens": 13,
        "sessionOutputTokensText": "13",
        "sessionTotalTokens": 43,
        "sessionHasCost": True,
        "sessionCostUsd": 0.0125,
        "sessionCostUsdText": "0.0125",
        "sessionTokenBadgeText": "43 ток",
    }
