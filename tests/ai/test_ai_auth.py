import pytest
from django.contrib.auth import get_user_model
from django.template.loader import get_template


@pytest.mark.django_db
def test_login_page_uses_english_authentication_and_guest_navigation_copy(client) -> None:
    response = client.get("/accounts/login/")

    assert response.status_code == 200
    content = response.content.decode()
    assert "Sign in | Dux" in content
    assert "Smart data assistant" in content
    assert "Enter your password" in content
    assert "Guest" in content
    assert "Not signed in" in content
    assert "Theme" in content
    assert 'aria-label="Toggle menu"' in content


def test_signup_template_uses_english_copy_and_neutral_placeholder() -> None:
    source = get_template("account/signup.html").template.source

    assert "Sign up | Dux" in source
    assert "Create an account to access Dux Chat" in source
    assert 'placeholder="name@company.com"' in source
    assert "Confirm password" in source
    assert "Already have an account?" in source


@pytest.mark.django_db
def test_authenticated_chat_navigation_uses_english_copy(client) -> None:
    user = get_user_model().objects.create_user(
        username="nav-test",
        email="nav-test@example.com",
        password="test-password",
    )
    client.force_login(user)

    response = client.get("/")

    assert response.status_code == 200
    content = response.content.decode()
    assert "New chat" in content
    assert "Theme" in content
    assert "Sign out" in content
    assert 'aria-label="Toggle menu"' in content
    assert "latest year represented in Chinook" in content
    assert "Which tracks appear in the most playlists?" in content
    assert content.count("using the bundled Chinook data") == 2
    assert "top 10 products by sales quantity" not in content


@pytest.mark.parametrize("path", ["/", "/run_chat/"])
@pytest.mark.django_db
def test_ai_routes_require_login(client, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 302
    assert response["Location"].startswith("/accounts/login/")
