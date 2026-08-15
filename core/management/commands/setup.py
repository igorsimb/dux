from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from core.username_utils import generate_unique_username

DEFAULT_ADMIN_EMAIL = "admin@test.com"
DEFAULT_ADMIN_PASSWORD = "password321"


class Command(BaseCommand):
    help = "Apply migrations and create the initial administrator account."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--noinput",
            "--no-input",
            action="store_false",
            dest="interactive",
            help="Create the development administrator without prompting.",
        )

    def handle(self, *args, **options) -> None:
        self.stdout.write("Applying database migrations...")
        call_command("migrate", interactive=False, verbosity=options["verbosity"])

        user_model = get_user_model()
        if user_model.objects.filter(is_superuser=True).exists():
            self.stdout.write(self.style.SUCCESS("An administrator already exists. Setup is complete."))
            return

        if options["interactive"]:
            self.stdout.write("No administrator exists. Create one now.")
            call_command("createsuperuser")
        else:
            self._create_development_superuser(user_model)

        self.stdout.write(self.style.SUCCESS("Setup is complete."))

    def _create_development_superuser(self, user_model) -> None:
        if user_model.objects.filter(email__iexact=DEFAULT_ADMIN_EMAIL).exists():
            raise CommandError(f"A non-superuser account already uses {DEFAULT_ADMIN_EMAIL}.")

        username_field = user_model._meta.get_field("username")
        username = generate_unique_username(
            email=DEFAULT_ADMIN_EMAIL,
            username_exists=lambda candidate: user_model.objects.filter(username=candidate).exists(),
            max_length=username_field.max_length,
        )
        user_model.objects.create_superuser(
            username=username,
            email=DEFAULT_ADMIN_EMAIL,
            password=DEFAULT_ADMIN_PASSWORD,
        )
        self.stdout.write(self.style.WARNING(f"Created development administrator {DEFAULT_ADMIN_EMAIL}."))
