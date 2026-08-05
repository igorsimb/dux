"""Shared LangGraph checkpointer for this Python process.

The in-memory saver lets separate HTTP requests handled by the same process
reuse thread state, but it does not survive process restart.

It is not shared across multiple Django worker processes.

This module also exposes cleanup hooks so tests, development workflows, and
future session-lifecycle integration can drop or reset in-memory thread state
explicitly when needed.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

_CHECKPOINTER: InMemorySaver | None = None


def get_checkpointer() -> InMemorySaver:
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER


def delete_thread_checkpoints(thread_id: str) -> None:
    get_checkpointer().delete_thread(thread_id)


def reset_checkpointer() -> None:
    global _CHECKPOINTER
    _CHECKPOINTER = None
