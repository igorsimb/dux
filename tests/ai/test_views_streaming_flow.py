from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from django.core.exceptions import SynchronousOnlyOperation
from django.utils.functional import SimpleLazyObject

from ai import views
from ai.ai_utils.logging_config import build_public_conversation_code
from ai.ai_utils.chat_session import build_thread_id


def make_request(
    *, user_id: int = 17, username: str = "i.dolgikh", session_key: str | None = None
) -> SimpleNamespace:
    class FakeSession:
        def __init__(self, initial_session_key: str | None) -> None:
            self.session_key = initial_session_key

        def save(self) -> None:
            if self.session_key is None:
                self.session_key = "django-session-1234"

    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, username=username),
        session=FakeSession(session_key),
    )


def test_run_chat_derives_stable_thread_id_and_keeps_robot_id_separate(
    monkeypatch,
) -> None:
    request = make_request()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {"userInput": "здарова", "chatSessionKey": "chat-tab-1234"},
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
        captured.update(
            {
                "user_text": user_text,
                "robot_id": robot_id,
                "thread_id": thread_id,
                "stream_timeout_seconds": stream_timeout_seconds,
                "session_key": request.session.session_key,
            }
        )
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> list[dict[str, object]]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        events = [await anext(stream), await anext(stream), await anext(stream)]
        await stream.aclose()
        return events

    events = asyncio.run(exercise())

    assert request.session.session_key is not None
    assert captured["user_text"] == "здарова"
    assert captured["stream_timeout_seconds"] == 360
    assert captured["thread_id"] == build_thread_id(
        user_id=17,
        session_key=request.session.session_key,
        client_key="chat-tab-1234",
    )
    assert captured["robot_id"] != captured["thread_id"]
    assert events[0] == {
        "kind": "signals",
        "payload": {
            "isWaitingResponse": True,
            "thinkingSeconds": 0,
            "UserFacingProgressMessage": "думает...",
            "chatSessionKey": "chat-tab-1234",
        },
    }


def test_run_chat_resolves_lazy_user_id_without_sync_only_error(monkeypatch) -> None:
    class AsyncUnsafeUser:
        @property
        def id(self) -> int:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return 17
            raise SynchronousOnlyOperation("sync user lookup in async context")

        @property
        def username(self) -> str:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return "i.dolgikh"
            raise SynchronousOnlyOperation("sync user lookup in async context")

    request = make_request()
    request.user = SimpleLazyObject(lambda: AsyncUnsafeUser())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "здарова",
            "chatSessionKey": "chat-tab-lazy-user",
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
        captured["thread_id"] = thread_id
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> None:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        await anext(stream)
        await stream.aclose()

    asyncio.run(exercise())

    assert captured["thread_id"] == build_thread_id(
        user_id=17,
        session_key=str(request.session.session_key),
        client_key="chat-tab-lazy-user",
    )


def test_run_chat_normalizes_invalid_chat_session_key_and_returns_safe_value(
    monkeypatch,
) -> None:
    request = make_request(user_id=23, session_key="django-session-invalid")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {"userInput": "здарова", "chatSessionKey": "   "},
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
        views, "normalize_chat_session_key", lambda raw_value: "chat-generated-safe"
    )
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def fake_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        captured.update({"thread_id": thread_id, "robot_id": robot_id})
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> dict[str, object]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        first_event = await anext(stream)
        await stream.aclose()
        return first_event

    first_event = asyncio.run(exercise())

    assert captured["thread_id"] == build_thread_id(
        user_id=23,
        session_key="django-session-invalid",
        client_key="chat-generated-safe",
    )
    assert captured["robot_id"] != captured["thread_id"]
    assert first_event == {
        "kind": "signals",
        "payload": {
            "isWaitingResponse": True,
            "thinkingSeconds": 0,
            "UserFacingProgressMessage": "думает...",
            "chatSessionKey": "chat-generated-safe",
        },
    }


def test_run_chat_same_client_key_still_isolates_thread_id_by_user_and_session(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {"userInput": "здарова", "chatSessionKey": "chat-shared-1234"},
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
        captured.append({"thread_id": thread_id, "robot_id": robot_id})
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise(request) -> None:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(request))
        await anext(stream)
        await stream.aclose()

    asyncio.run(exercise(make_request(user_id=31, session_key="django-session-a")))
    asyncio.run(exercise(make_request(user_id=32, session_key="django-session-b")))

    assert len(captured) == 2
    assert captured[0]["thread_id"] == build_thread_id(
        user_id=31,
        session_key="django-session-a",
        client_key="chat-shared-1234",
    )
    assert captured[1]["thread_id"] == build_thread_id(
        user_id=32,
        session_key="django-session-b",
        client_key="chat-shared-1234",
    )
    assert captured[0]["thread_id"] != captured[1]["thread_id"]


def test_finalize_producer_task_cancels_running_chat_session() -> None:
    finished = asyncio.Event()

    async def blocking_session() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            finished.set()
            raise

    async def exercise_cleanup() -> bool:
        producer_task = asyncio.create_task(blocking_session())
        await asyncio.sleep(0)
        await views.finalize_producer_task(producer_task)
        return finished.is_set() and producer_task.cancelled()

    assert asyncio.run(exercise_cleanup())


def test_finalize_producer_task_logs_cancellation(monkeypatch) -> None:
    debug_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        views.logger,
        "debug",
        lambda message, *args: debug_calls.append((str(message), args)),
    )

    async def blocking_session() -> None:
        await asyncio.Event().wait()

    async def exercise_cleanup() -> list[tuple[str, tuple[object, ...]]]:
        producer_task = asyncio.create_task(blocking_session())
        await asyncio.sleep(0)
        await views.finalize_producer_task(producer_task)
        return debug_calls

    logged = asyncio.run(exercise_cleanup())
    messages = [message for message, _args in logged]

    assert any("producer_cancelling" in message for message in messages)
    assert any("producer_cancelled" in message for message in messages)


def test_run_chat_sets_waiting_response_true_in_initial_signal(monkeypatch) -> None:
    request = make_request()

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {"userInput": "здарова", "chatSessionKey": "chat-tab-5678"},
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

    async def fake_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def exercise() -> dict[str, object]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        first_event = await anext(stream)
        await stream.aclose()
        return first_event

    first_event = asyncio.run(exercise())

    assert first_event == {
        "kind": "signals",
        "payload": {
            "isWaitingResponse": True,
            "thinkingSeconds": 0,
            "UserFacingProgressMessage": "думает...",
            "chatSessionKey": "chat-tab-5678",
        },
    }


def test_run_chat_logs_final_visible_assistant_reply(monkeypatch) -> None:
    request = make_request()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {"userInput": "здарова", "chatSessionKey": "chat-tab-log-1234"},
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
    monkeypatch.setattr(
        views,
        "append_chat_log_turn",
        lambda username, thread_id, user_text, ai_text: captured.update(
            {
                "username": username,
                "thread_id": thread_id,
                "user_text": user_text,
                "ai_text": ai_text,
            }
        ),
    )
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def fake_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        await queue.put({"kind": "token", "text": "Здравствуйте"})
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> list[object]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        events = [event async for event in stream]
        return events

    events = asyncio.run(exercise())

    assert any(
        isinstance(event, dict) and event.get("kind") == "robot-text"
        for event in events
    )
    assert captured == {
        "username": request.user.username,
        "thread_id": build_thread_id(
            user_id=17,
            session_key=str(request.session.session_key),
            client_key="chat-tab-log-1234",
        ),
        "user_text": "здарова",
        "ai_text": "Здравствуйте",
    }


def test_run_chat_renders_structured_blocks_and_logs_visible_summary(monkeypatch) -> None:
    request = make_request(session_key="django-session-blocks")
    request.user.has_perm = lambda permission: permission == "core.view_answer_notes"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {"userInput": "топ клиентов", "chatSessionKey": "chat-tab-blocks"},
    )
    monkeypatch.setattr(views, "DatastarResponse", lambda generator: generator)
    monkeypatch.setattr(views, "append_user_message", lambda text: {"kind": "user", "text": text})
    monkeypatch.setattr(
        views,
        "append_robot_container",
        lambda robot_id: {"kind": "robot-container", "robot_id": robot_id},
    )
    monkeypatch.setattr(
        views,
        "append_robot_blocks",
        lambda robot_id, blocks, **kwargs: {
            "kind": "robot-blocks",
            "robot_id": robot_id,
            "blocks": blocks,
            "permissions": kwargs,
        },
    )
    monkeypatch.setattr(
        views,
        "append_chat_log_turn",
        lambda username, thread_id, user_text, ai_text: captured.update(
            {"username": username, "thread_id": thread_id, "user_text": user_text, "ai_text": ai_text}
        ),
    )
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    blocks = [
        {"id": "c1", "type": "commentary", "format": "markdown", "content": "Вот результат."},
        {
            "id": "sql-result-1",
            "type": "data_table",
            "title": "Топ клиентов",
            "columns": [],
            "rows": [],
            "meta": {"row_count": 10, "rendered_row_count": 5, "truncated": True},
        },
    ]

    async def fake_session(*, user_text, robot_id, thread_id, queue, stream_timeout_seconds) -> None:
        await queue.put({"kind": "blocks", "blocks": blocks})
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> list[object]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        return [event async for event in stream]

    events = asyncio.run(exercise())

    robot_container = next(
        event for event in events if isinstance(event, dict) and event.get("kind") == "robot-container"
    )
    assert {
        "kind": "robot-blocks",
        "robot_id": robot_container["robot_id"],
        "blocks": blocks,
        "permissions": {"can_view_answer_notes": True, "can_view_raw_sql": False},
    } in events
    assert captured["ai_text"] == (
        "Вот результат.\n\n"
        "[Таблица: Топ клиентов, строк показано: 5 из 10]"
    )


def test_run_chat_smalltalk_meta_streams_short_domain_steering_reply(
    monkeypatch,
) -> None:
    request = make_request(session_key="django-session-smalltalk")

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "что ты умеешь?",
            "chatSessionKey": "chat-tab-smalltalk",
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
        await queue.put(
            {
                "kind": "token",
                "text": "Могу помочь с SQL-запросами, продажами, заказами, клиентами и остатками.",
            }
        )
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> list[object]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        return [event async for event in stream]

    events = asyncio.run(exercise())

    assert any(
        isinstance(event, dict)
        and event.get("kind") == "robot-text"
        and "SQL" in str(event.get("text"))
        and "заказами" in str(event.get("text"))
        and "клиентами" in str(event.get("text")).lower()
        for event in events
    )


def test_run_chat_timeout_emits_timeout_message_once_and_resets_waiting_state(
    monkeypatch,
) -> None:
    request = make_request(session_key="django-session-timeout")

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {"userInput": "здарова", "chatSessionKey": "chat-tab-9012"},
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
    monkeypatch.setattr(views, "build_timeout_message", lambda: "TIMEOUT")
    monkeypatch.setattr(views, "append_chat_log_turn", lambda *args, **kwargs: None)

    async def blocking_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(views, "run_chat_session", blocking_session)
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def exercise() -> list[object]:
        real_loop = asyncio.get_running_loop()

        class FakeLoop:
            def __init__(self) -> None:
                self.values = iter([0.0, 0.0, 1201.0])

            def time(self) -> float:
                return next(self.values)

            def run_in_executor(self, executor, func, *args):
                return real_loop.run_in_executor(executor, func, *args)

        monkeypatch.setattr(views.asyncio, "get_running_loop", lambda: FakeLoop())
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        events: list[object] = []
        async for event in stream:
            events.append(event)
        return events

    events = asyncio.run(exercise())

    timeout_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("kind") == "robot-text"
    ]
    reset_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("kind") == "signals"
        and event.get("payload", {}).get("isWaitingResponse") is False
    ]

    assert len(timeout_events) == 1
    assert timeout_events[0]["kind"] == "robot-text"
    assert timeout_events[0]["robot_id"]
    assert str(timeout_events[0]["text"]).startswith("TIMEOUT")
    assert len(reset_events) == 1


def test_run_chat_timeout_message_includes_public_conversation_code(
    monkeypatch,
) -> None:
    request = make_request(session_key="django-session-timeout")
    captured_chat_log: dict[str, object] = {}

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "здарова",
            "chatSessionKey": "chat-tab-timeout-code",
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
    monkeypatch.setattr(
        views,
        "append_chat_log_turn",
        lambda username, thread_id, user_text, ai_text: captured_chat_log.update(
            {
                "username": username,
                "thread_id": thread_id,
                "user_text": user_text,
                "ai_text": ai_text,
            }
        ),
    )

    async def blocking_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(views, "run_chat_session", blocking_session)
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def exercise() -> list[object]:
        real_loop = asyncio.get_running_loop()

        class FakeLoop:
            def __init__(self) -> None:
                self.values = iter([0.0, 0.0, 1201.0])

            def time(self) -> float:
                return next(self.values)

            def run_in_executor(self, executor, func, *args):
                return real_loop.run_in_executor(executor, func, *args)

        monkeypatch.setattr(views.asyncio, "get_running_loop", lambda: FakeLoop())
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        return [event async for event in stream]

    events = asyncio.run(exercise())

    timeout_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("kind") == "robot-text"
    ]
    expected_thread_id = build_thread_id(
        user_id=17,
        session_key="django-session-timeout",
        client_key="chat-tab-timeout-code",
    )
    conversation_code = build_public_conversation_code(expected_thread_id)

    assert len(timeout_events) == 1
    assert conversation_code in str(timeout_events[0].get("text"))
    assert captured_chat_log["thread_id"] == expected_thread_id
    assert conversation_code not in str(captured_chat_log["ai_text"])


def test_run_chat_logs_thread_id_when_cleanup_task_fails(monkeypatch) -> None:
    request = make_request()
    exception_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "здарова",
            "chatSessionKey": "chat-tab-cleanup-log",
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
    monkeypatch.setattr(
        views.logger,
        "exception",
        lambda message, *args: exception_calls.append((str(message), args)),
    )

    async def fake_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        await queue.put({"kind": "token", "text": "Здравствуйте"})
        await queue.put(None)

    async def failing_finalize(_producer_task) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(views, "run_chat_session", fake_session)
    monkeypatch.setattr(views, "finalize_producer_task", failing_finalize)

    async def exercise() -> None:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        async for _event in stream:
            pass

    asyncio.run(exercise())

    expected_thread_id = build_thread_id(
        user_id=17,
        session_key=str(request.session.session_key),
        client_key="chat-tab-cleanup-log",
    )
    assert len(exception_calls) == 1
    message, args = exception_calls[0]
    assert args == ()
    assert "cleanup_failed" in message
    assert build_public_conversation_code(expected_thread_id)[:8] in message


def test_run_chat_logs_timeout_text_when_that_is_final_visible_reply(
    monkeypatch,
) -> None:
    request = make_request(session_key="django-session-timeout")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "здарова",
            "chatSessionKey": "chat-tab-timeout-log",
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
    monkeypatch.setattr(views, "build_timeout_message", lambda **kwargs: "TIMEOUT")
    monkeypatch.setattr(
        views,
        "append_chat_log_turn",
        lambda username, thread_id, user_text, ai_text: captured.update(
            {
                "username": username,
                "thread_id": thread_id,
                "user_text": user_text,
                "ai_text": ai_text,
            }
        ),
    )
    monkeypatch.setattr(
        views.SSE,
        "patch_signals",
        staticmethod(lambda payload: {"kind": "signals", "payload": payload}),
    )

    async def blocking_session(
        *, user_text, robot_id, thread_id, queue, stream_timeout_seconds
    ) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(views, "run_chat_session", blocking_session)

    async def exercise() -> None:
        real_loop = asyncio.get_running_loop()

        class FakeLoop:
            def __init__(self) -> None:
                self.values = iter([0.0, 0.0, 1201.0])

            def time(self) -> float:
                return next(self.values)

            def run_in_executor(self, executor, func, *args):
                return real_loop.run_in_executor(executor, func, *args)

        monkeypatch.setattr(views.asyncio, "get_running_loop", lambda: FakeLoop())
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        async for _event in stream:
            pass

    asyncio.run(exercise())

    assert captured == {
        "username": request.user.username,
        "thread_id": build_thread_id(
            user_id=17,
            session_key="django-session-timeout",
            client_key="chat-tab-timeout-log",
        ),
        "user_text": "здарова",
        "ai_text": "TIMEOUT",
    }


def test_run_chat_theme_change_emits_theme_signal_and_short_acknowledgement(
    monkeypatch,
) -> None:
    request = make_request(session_key="django-session-theme")

    monkeypatch.setattr(
        views,
        "read_signals",
        lambda request: {
            "userInput": "смени тему на nord",
            "chatSessionKey": "chat-tab-theme",
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
        await queue.put({"kind": "token", "text": "Готово, переключил тему на nord."})
        await queue.put(None)

    monkeypatch.setattr(views, "run_chat_session", fake_session)

    async def exercise() -> list[object]:
        run_chat_view = getattr(views.run_chat, "__wrapped__", views.run_chat)
        stream = cast(Any, await run_chat_view(cast(Any, request)))
        return [event async for event in stream]

    events = asyncio.run(exercise())

    assert {"kind": "signals", "payload": {"ThemeSwitchCommand": "nord"}} in events
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "robot-text"
        and event.get("text") == "Готово, переключил тему на nord."
        for event in events
    )


def test_ai_main_template_reuses_session_key_on_refresh_and_generates_new_key_for_new_tab() -> (
    None
):
    ai_content = (
        Path(__file__).resolve().parents[2] / "ai" / "templates" / "ai" / "ai_main.html"
    ).read_text(encoding="utf-8")
    navbar_content = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "templates"
        / "core"
        / "navbar.html"
    ).read_text(encoding="utf-8")
    assert "data-signals:chat-session-key__ifmissing=" in ai_content
    assert "sessionStorage.getItem('chatSessionKey')" in ai_content
    assert "window.aiChatSessionUi.newChatSessionKey()" in ai_content
    assert 'typeof globalThis.crypto.randomUUID === "function"' in ai_content
    assert 'typeof globalThis.crypto.getRandomValues === "function"' in ai_content
    assert (
        "chatApp.dispatchEvent(new CustomEvent('ai-new-chat-requested', { bubbles: true }));"
        in navbar_content
    )


def test_ai_main_template_help_text_matches_short_term_memory_model() -> None:
    content = (
        Path(__file__).resolve().parents[2] / "ai" / "templates" / "ai" / "ai_main.html"
    ).read_text(encoding="utf-8")

    assert (
        "Робот помнит текущий разговор в рамках активной вкладки браузера." in content
    )
