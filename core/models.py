from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    class Meta:
        permissions = [
            ("view_answer_notes", "Can view AI answer detail notes"),
            ("view_raw_sql", "Can view raw SQL in AI answers"),
        ]

    def __str__(self):
        return self.first_name or self.username
