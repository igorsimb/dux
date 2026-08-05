import pytest
from django.contrib.auth.models import Group, Permission


@pytest.mark.django_db
def test_answer_details_permissions_exist() -> None:
    assert Permission.objects.filter(content_type__app_label="core", codename="view_answer_notes").exists()
    assert Permission.objects.filter(content_type__app_label="core", codename="view_raw_sql").exists()


@pytest.mark.django_db
def test_answer_details_groups_include_expected_permissions() -> None:
    answer_notes_group = Group.objects.get(name="AI answer notes viewers")
    raw_sql_group = Group.objects.get(name="AI raw SQL viewers")

    assert answer_notes_group.permissions.filter(content_type__app_label="core", codename="view_answer_notes").exists()
    assert raw_sql_group.permissions.filter(content_type__app_label="core", codename="view_raw_sql").exists()


@pytest.mark.django_db
def test_answer_details_group_permissions_grant_user_access(django_user_model) -> None:
    user = django_user_model.objects.create_user(username="viewer", password="password")
    user.groups.add(Group.objects.get(name="AI answer notes viewers"))

    assert user.has_perm("core.view_answer_notes")
    assert not user.has_perm("core.view_raw_sql")

    user.groups.add(Group.objects.get(name="AI raw SQL viewers"))
    for cache_name in ("_perm_cache", "_user_perm_cache", "_group_perm_cache"):
        if hasattr(user, cache_name):
            delattr(user, cache_name)

    assert user.has_perm("core.view_answer_notes")
    assert user.has_perm("core.view_raw_sql")
