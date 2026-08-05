from core.username_utils import generate_unique_username


def test_generate_unique_username_uses_email_local_part_when_available():
    username = generate_unique_username(
        email="alice@example.com", username_exists=lambda candidate: False
    )

    assert username == "alice"


def test_generate_unique_username_normalizes_and_adds_suffix_when_taken():
    taken = {"john-smith", "john-smith-1", "john-smith-2"}

    username = generate_unique_username(
        email="John Smith@example.com",
        username_exists=lambda candidate: candidate in taken,
    )

    assert username == "john-smith-3"


def test_generate_unique_username_falls_back_when_local_part_is_empty_after_normalization():
    username = generate_unique_username(
        email="...@example.com", username_exists=lambda candidate: False
    )

    assert username == "user"
