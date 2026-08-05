import pytest


@pytest.mark.parametrize("path", ["/", "/run_chat/"])
@pytest.mark.django_db
def test_ai_routes_require_login(client, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 302
    assert response["Location"].startswith("/accounts/login/")
