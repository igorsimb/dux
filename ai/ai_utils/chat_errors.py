"""Error and timeout message builders for chat streaming flows."""

from __future__ import annotations


def build_error_message(exc: BaseException | None = None) -> str:
    """Build the base error text used for logs, transcripts, and UI output."""
    message = "\n\n(Ошибка: не удалось получить ответ. Попробуйте запрос еще раз.)"
    if exc is None:
        return message

    message_text = str(exc)
    if "unsupported_country_region_territory" in message_text:
        return "\n\n(Ошибка: OpenAI недоступен из этого региона. Проверьте прокси.)"
    if "OPENAI_API_KEY" in message_text:
        return "\n\n(Ошибка: не задан OPENAI_API_KEY.)"
    return message


def build_timeout_message() -> str:
    """Build the base timeout text used for logs, transcripts, and UI output."""
    return "\n\n(Ошибка: истекло время ожидания. Запрос отменен.)"


def build_user_facing_message(message: str, conversation_code: str) -> str:
    """Append the short support code to a UI-visible message when available."""
    code = str(conversation_code).strip()
    if not code:
        return str(message)
    return f"{message} Код разговора: {code}."
