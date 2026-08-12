from __future__ import annotations

import asyncio

from ai.ai_utils import chat_runtime
from ai.ai_utils.logging_config import build_public_conversation_code


class _MissingToolResponseError(RuntimeError):
    pass


def test_run_chat_session_enqueues_error_token_and_terminal_none(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_runtime,
        "build_model_and_tools",
        lambda stream_timeout_seconds: (_ for _ in ()).throw(ValueError("bad config")),
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        await chat_runtime.run_chat_session(
            user_text="test",
            robot_id="robot-async-1",
            thread_id="thread-async-1",
            queue=queue,
            stream_timeout_seconds=10,
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert len(items) == 2
    first_item = items[0]
    assert isinstance(first_item, dict)
    assert first_item.get("kind") == "token"
    assert "Error" in str(first_item.get("text"))
    assert build_public_conversation_code("thread-async-1") not in str(
        first_item.get("text")
    )
    assert build_public_conversation_code("thread-async-1") in str(
        first_item.get("display_text")
    )
    assert items[-1] is None


def test_run_chat_session_passes_explicit_thread_id_to_streaming(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        chat_runtime,
        "build_model_and_tools",
        lambda stream_timeout_seconds: ("model", ["tool"], "gpt-test"),
    )
    monkeypatch.setattr(chat_runtime, "build_chat_agent", lambda model, tools: "agent")

    async def fake_stream(agent, user_text, robot_id, thread_id, queue, *, model_name) -> None:
        captured.update(
            {
                "agent": agent,
                "user_text": user_text,
                "robot_id": robot_id,
                "thread_id": thread_id,
                "model_name": model_name,
            }
        )

    monkeypatch.setattr(chat_runtime, "produce_agent_stream_async", fake_stream)

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        await chat_runtime.run_chat_session(
            user_text="test",
            robot_id="robot-async-1",
            thread_id="thread-async-1",
            queue=queue,
            stream_timeout_seconds=10,
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert captured == {
        "agent": "agent",
        "user_text": "test",
        "robot_id": "robot-async-1",
        "thread_id": "thread-async-1",
        "model_name": "gpt-test",
    }
    assert items == [None]


def test_run_chat_session_resets_poisoned_thread_and_retries_once(monkeypatch) -> None:
    attempts: list[str] = []
    deleted_threads: list[str] = []

    async def fake_run_once(
        *,
        user_text: str,
        robot_id: str,
        thread_id: str,
        queue: asyncio.Queue[object | None],
        stream_timeout_seconds: int,
    ) -> None:
        attempts.append(thread_id)
        if len(attempts) == 1:
            raise _MissingToolResponseError(
                "An assistant message with 'tool_calls' must be followed by tool messages responding to each "
                "'tool_call_id'."
            )
        await queue.put({"kind": "token", "text": "ok"})

    monkeypatch.setattr(chat_runtime, "_run_chat_session_once", fake_run_once)
    monkeypatch.setattr(
        chat_runtime,
        "delete_thread_checkpoints",
        lambda thread_id: deleted_threads.append(thread_id),
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        await chat_runtime.run_chat_session(
            user_text="test",
            robot_id="robot-async-1",
            thread_id="thread-poisoned-1",
            queue=queue,
            stream_timeout_seconds=10,
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert attempts == ["thread-poisoned-1", "thread-poisoned-1"]
    assert deleted_threads == ["thread-poisoned-1"]
    assert items == [{"kind": "token", "text": "ok"}, None]


def test_run_chat_session_uses_retry_error_when_reset_retry_also_fails(
    monkeypatch,
) -> None:
    deleted_threads: list[str] = []

    async def fake_run_once(
        *,
        user_text: str,
        robot_id: str,
        thread_id: str,
        queue: asyncio.Queue[object | None],
        stream_timeout_seconds: int,
    ) -> None:
        if not deleted_threads:
            raise _MissingToolResponseError(
                "An assistant message with 'tool_calls' must be followed by tool messages responding to each "
                "'tool_call_id'."
            )
        raise ValueError("retry failed")

    monkeypatch.setattr(chat_runtime, "_run_chat_session_once", fake_run_once)
    monkeypatch.setattr(
        chat_runtime,
        "delete_thread_checkpoints",
        lambda thread_id: deleted_threads.append(thread_id),
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        await chat_runtime.run_chat_session(
            user_text="test",
            robot_id="robot-async-1",
            thread_id="thread-poisoned-2",
            queue=queue,
            stream_timeout_seconds=10,
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert deleted_threads == ["thread-poisoned-2"]
    assert len(items) == 2
    first_item = items[0]
    assert isinstance(first_item, dict)
    assert first_item.get("kind") == "token"
    assert "Error" in str(first_item.get("text"))
    assert items[-1] is None


def test_run_chat_session_enqueues_error_when_thread_reset_fails(monkeypatch) -> None:
    async def fake_run_once(
        *,
        user_text: str,
        robot_id: str,
        thread_id: str,
        queue: asyncio.Queue[object | None],
        stream_timeout_seconds: int,
    ) -> None:
        raise _MissingToolResponseError(
            "An assistant message with 'tool_calls' must be followed by tool messages responding to each "
            "'tool_call_id'."
        )

    monkeypatch.setattr(chat_runtime, "_run_chat_session_once", fake_run_once)
    monkeypatch.setattr(
        chat_runtime,
        "delete_thread_checkpoints",
        lambda thread_id: (_ for _ in ()).throw(RuntimeError("reset failed")),
    )

    async def exercise() -> list[object]:
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        await chat_runtime.run_chat_session(
            user_text="test",
            robot_id="robot-async-1",
            thread_id="thread-poisoned-3",
            queue=queue,
            stream_timeout_seconds=10,
        )
        items: list[object] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(exercise())

    assert len(items) == 2
    first_item = items[0]
    assert isinstance(first_item, dict)
    assert first_item.get("kind") == "token"
    assert "Error" in str(first_item.get("text"))
    assert build_public_conversation_code("thread-poisoned-3") not in str(
        first_item.get("text")
    )
    assert build_public_conversation_code("thread-poisoned-3") in str(
        first_item.get("display_text")
    )
    assert items[-1] is None


def test_run_chat_session_logs_thread_id_on_exception_paths(monkeypatch) -> None:
    exception_calls: list[tuple[str, tuple[object, ...]]] = []

    monkeypatch.setattr(
        chat_runtime,
        "build_model_and_tools",
        lambda stream_timeout_seconds: (_ for _ in ()).throw(ValueError("bad config")),
    )
    monkeypatch.setattr(
        chat_runtime.logger,
        "exception",
        lambda message, *args: exception_calls.append((str(message), args)),
    )

    async def exercise() -> None:
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        await chat_runtime.run_chat_session(
            user_text="test",
            robot_id="robot-async-1",
            thread_id="thread-async-1",
            queue=queue,
            stream_timeout_seconds=10,
        )

    asyncio.run(exercise())

    assert len(exception_calls) == 1
    message, args = exception_calls[0]
    assert args == ()
    assert "failed" in message
    assert "thread-a" in message
