"""Error and timeout message builders for chat streaming flows."""

from __future__ import annotations


def build_error_message(exc: BaseException | None = None) -> str:
    """Build the base error text used for logs, transcripts, and UI output."""
    message = "\n\n(Error: Could not get a response. Please try again.)"
    if exc is None:
        return message

    message_text = str(exc)
    if "unsupported_country_region_territory" in message_text:
        return "\n\n(Error: OpenAI is unavailable in this region. Check your proxy.)"
    if "OPENAI_API_KEY" in message_text:
        return "\n\n(Error: OPENAI_API_KEY is not set.)"
    return message


def build_timeout_message() -> str:
    """Build the base timeout text used for logs, transcripts, and UI output."""
    return "\n\n(Error: The request timed out and was canceled.)"


def build_user_facing_message(message: str, conversation_code: str) -> str:
    """Append the short support code to a UI-visible message when available."""
    code = str(conversation_code).strip()
    if not code:
        return str(message)
    return f"{message} Conversation code: {code}."
