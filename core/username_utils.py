from collections.abc import Callable

from django.utils.text import slugify


def generate_unique_username(
    email: str, username_exists: Callable[[str], bool], max_length: int = 150
) -> str:
    """Build a unique slugified username from an email local part."""
    local_part = (email or "").split("@", 1)[0]
    base_username = slugify(local_part) or "user"
    base_username = base_username[:max_length].strip("-") or "user"
    candidate_username = base_username
    suffix_number = 1

    while username_exists(candidate_username):
        numeric_suffix = f"-{suffix_number}"
        truncated_base_username = base_username[: max_length - len(numeric_suffix)].rstrip("-") or "user"
        candidate_username = f"{truncated_base_username}{numeric_suffix}"
        suffix_number += 1

    return candidate_username
