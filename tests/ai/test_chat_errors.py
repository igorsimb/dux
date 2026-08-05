import pytest

from ai.ai_utils.chat_errors import (
    build_error_message,
    build_timeout_message,
    build_user_facing_message,
)


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (None, "\n\n(Ошибка: не удалось получить ответ. Попробуйте запрос еще раз.)"),
        (
            RuntimeError("provider unavailable"),
            "\n\n(Ошибка: не удалось получить ответ. Попробуйте запрос еще раз.)",
        ),
        (
            RuntimeError("unsupported_country_region_territory"),
            "\n\n(Ошибка: OpenAI недоступен из этого региона. Проверьте прокси.)",
        ),
        (
            RuntimeError("Missing OPENAI_API_KEY"),
            "\n\n(Ошибка: не задан OPENAI_API_KEY.)",
        ),
    ],
)
def test_build_error_message_returns_public_error_text(
    error: BaseException | None, expected_message: str
) -> None:
    assert build_error_message(error) == expected_message


def test_build_timeout_message_returns_public_timeout_text() -> None:
    assert build_timeout_message() == "\n\n(Ошибка: истекло время ожидания. Запрос отменен.)"


def test_build_user_facing_message_appends_normalized_conversation_code() -> None:
    assert build_user_facing_message("Ошибка.", "  ab12cd34  ") == "Ошибка. Код разговора: ab12cd34."


def test_build_user_facing_message_omits_blank_conversation_code() -> None:
    assert build_user_facing_message("Ошибка.", "   ") == "Ошибка."
