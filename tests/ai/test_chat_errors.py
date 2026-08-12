import pytest

from ai.ai_utils.chat_errors import (
    build_error_message,
    build_timeout_message,
    build_user_facing_message,
)


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (None, "\n\n(Error: Could not get a response. Please try again.)"),
        (
            RuntimeError("provider unavailable"),
            "\n\n(Error: Could not get a response. Please try again.)",
        ),
        (
            RuntimeError("unsupported_country_region_territory"),
            "\n\n(Error: OpenAI is unavailable in this region. Check your proxy.)",
        ),
        (
            RuntimeError("Missing OPENAI_API_KEY"),
            "\n\n(Error: OPENAI_API_KEY is not set.)",
        ),
    ],
)
def test_build_error_message_returns_public_error_text(
    error: BaseException | None, expected_message: str
) -> None:
    assert build_error_message(error) == expected_message


def test_build_timeout_message_returns_public_timeout_text() -> None:
    assert build_timeout_message() == "\n\n(Error: The request timed out and was canceled.)"


def test_build_user_facing_message_appends_normalized_conversation_code() -> None:
    assert build_user_facing_message("Error.", "  ab12cd34  ") == "Error. Conversation code: ab12cd34."


def test_build_user_facing_message_omits_blank_conversation_code() -> None:
    assert build_user_facing_message("Error.", "   ") == "Error."
