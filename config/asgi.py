"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

from ai.ai_utils import logging_config

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
logging_config.configure_logging()

application = get_asgi_application()
