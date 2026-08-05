"""Helpers for browser chat session keys and server-side thread ids.

The browser keeps a lightweight `chatSessionKey` so one tab can reuse the same
visible chat session across refreshes. The backend never trusts that client key
as the real conversation id; instead it derives a stable `thread_id` from the
authenticated user, Django session, and normalized client key.
"""

from __future__ import annotations

import re
import uuid
from hashlib import sha256

CLIENT_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]{8,128}$")
THREAD_ID_PREFIX = "chat-thread-"


def new_chat_session_key() -> str:
    return f"chat-{uuid.uuid4().hex}"


def normalize_chat_session_key(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if CLIENT_KEY_RE.fullmatch(value):
        return value
    return new_chat_session_key()


def build_thread_id(*, user_id: object, session_key: str, client_key: str) -> str:
    raw_value = f"user:{user_id}|session:{session_key}|chat:{client_key}"
    digest = sha256(raw_value.encode("utf-8")).hexdigest()
    return f"{THREAD_ID_PREFIX}{digest}"
