from allauth.account.adapter import DefaultAccountAdapter
from django.contrib.auth import get_user_model
from core.username_utils import generate_unique_username


class AccountAdapter(DefaultAccountAdapter):
    def is_open_for_signup(self, request) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return False

    def populate_username(self, request, user) -> None:
        if user.username:
            return

        user_model = get_user_model()
        username_field = user_model._meta.get_field("username")

        user.username = generate_unique_username(
            email=user.email,
            username_exists=lambda candidate: user_model.objects.filter(
                username=candidate
            ).exists(),
            max_length=username_field.max_length,
        )
