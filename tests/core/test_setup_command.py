from io import StringIO

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command


@pytest.mark.django_db
def test_setup_noinput_creates_default_superuser() -> None:
    output = StringIO()

    call_command("setup", interactive=False, stdout=output, verbosity=0)

    user = get_user_model().objects.get(email="admin@test.com")
    assert user.username == "admin"
    assert user.is_staff
    assert user.is_superuser
    assert user.check_password("password321")
    assert "Setup is complete." in output.getvalue()


@pytest.mark.django_db
def test_setup_does_not_replace_existing_superuser_password() -> None:
    user_model = get_user_model()
    existing_user = user_model.objects.create_superuser(
        username="owner",
        email="owner@example.com",
        password="keep-this-password",
    )
    output = StringIO()

    call_command("setup", interactive=False, stdout=output, verbosity=0)

    existing_user.refresh_from_db()
    assert existing_user.check_password("keep-this-password")
    assert user_model.objects.filter(is_superuser=True).count() == 1
    assert "An administrator already exists." in output.getvalue()
