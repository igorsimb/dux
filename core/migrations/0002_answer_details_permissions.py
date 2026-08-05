# Generated for answer details panel permissions.

from django.db import migrations, models


ANSWER_NOTES_GROUP = "AI answer notes viewers"
RAW_SQL_GROUP = "AI raw SQL viewers"

ANSWER_NOTES_PERMISSION = (
    "view_answer_notes",
    "Can view AI answer detail notes",
)
RAW_SQL_PERMISSION = (
    "view_raw_sql",
    "Can view raw SQL in AI answers",
)


def create_answer_details_groups(apps, schema_editor):
    ContentType = apps.get_model("contenttypes", "ContentType")
    Group = apps.get_model("auth", "Group")
    Permission = apps.get_model("auth", "Permission")

    user_content_type, _ = ContentType.objects.get_or_create(app_label="core", model="user")
    answer_notes_permission = Permission.objects.get_or_create(
        content_type=user_content_type,
        codename=ANSWER_NOTES_PERMISSION[0],
        defaults={"name": ANSWER_NOTES_PERMISSION[1]},
    )[0]
    raw_sql_permission = Permission.objects.get_or_create(
        content_type=user_content_type,
        codename=RAW_SQL_PERMISSION[0],
        defaults={"name": RAW_SQL_PERMISSION[1]},
    )[0]

    answer_notes_group = Group.objects.get_or_create(name=ANSWER_NOTES_GROUP)[0]
    answer_notes_group.permissions.add(answer_notes_permission)
    raw_sql_group = Group.objects.get_or_create(name=RAW_SQL_GROUP)[0]
    raw_sql_group.permissions.add(raw_sql_permission)


def remove_answer_details_groups(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    Permission = apps.get_model("auth", "Permission")

    Group.objects.filter(name__in=[ANSWER_NOTES_GROUP, RAW_SQL_GROUP]).delete()
    Permission.objects.filter(codename__in=[ANSWER_NOTES_PERMISSION[0], RAW_SQL_PERMISSION[0]]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="user",
            options={"permissions": [ANSWER_NOTES_PERMISSION, RAW_SQL_PERMISSION]},
        ),
        migrations.RunPython(create_answer_details_groups, remove_answer_details_groups),
    ]
