from __future__ import annotations

import importlib
import sys


def test_manage_main_configures_logging_before_running_command(monkeypatch) -> None:
    import manage

    calls: list[object] = []

    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.setattr(
        "ai.ai_utils.logging_config.configure_logging",
        lambda: calls.append("configure_logging"),
    )
    monkeypatch.setattr(
        "django.core.management.execute_from_command_line",
        lambda argv: calls.append(("execute_from_command_line", argv)),
    )

    manage.main()

    assert calls[0] == "configure_logging"
    assert calls[1] == ("execute_from_command_line", sys.argv)


def test_asgi_module_configures_logging_before_building_application(
    monkeypatch,
) -> None:
    calls: list[object] = []
    module_name = "config.asgi"

    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        "ai.ai_utils.logging_config.configure_logging",
        lambda: calls.append("configure_logging"),
    )
    monkeypatch.setattr(
        "django.core.asgi.get_asgi_application",
        lambda: calls.append("get_asgi_application") or object(),
    )

    imported_module = importlib.import_module(module_name)

    assert calls == ["configure_logging", "get_asgi_application"]
    assert imported_module.application is not None
