import config.settings as project_settings


def test_allauth_signup_fields_hide_username():
    assert project_settings.ACCOUNT_SIGNUP_FIELDS == [
        "email*",
        "password1*",
        "password2*",
    ]


def test_allauth_login_method_is_email_only():
    assert project_settings.ACCOUNT_LOGIN_METHODS == {"email"}
