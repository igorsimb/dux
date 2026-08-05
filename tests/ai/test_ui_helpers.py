from __future__ import annotations

import os
import re
from decimal import Decimal
from html import unescape

import django
from django.apps import apps
from django.conf import settings

from ai.ai_utils.ui import (
    format_table_cell_value,
    make_robot_blocks_html,
    make_robot_message_html,
    make_user_message_html,
)


def extract_chat_blocks_json(html: str) -> str:
    match = re.search(r'data-chat-blocks="([^"]*)"', html)
    assert match is not None
    return unescape(match.group(1))


def block_with_details() -> dict:
    return {
        "id": "t1",
        "type": "data_table",
        "title": "Продажи",
        "columns": [{"key": "customer", "label": "Клиент", "type": "string"}],
        "rows": [{"customer": "ООО Тест"}],
        "meta": {"row_count": 10, "rendered_row_count": 1, "truncated": True},
        "details": {
            "facts": {
                "source_id": "mssql_default",
                "dialect": "tsql",
                "validated_id": "vid-1",
                "tables": ["dbo.customer_orders"],
                "raw_sql": "SELECT * FROM dbo.customer_orders",
            },
            "notes": [
                {"label": "Источник :", "value": "model-chosen-source"},
                {"label": "Период", "value": "последние 30 дней"},
            ],
        },
    }


def configure_django_templates() -> None:
    if not settings.configured:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    if not apps.ready:
        django.setup()


def test_make_user_message_html_marks_persisted_user_messages() -> None:
    html = make_user_message_html("Привет")

    assert 'data-chat-persist="true"' in html
    assert 'data-chat-role="user"' in html


def test_make_robot_message_html_marks_persisted_assistant_messages() -> None:
    html = make_robot_message_html("robot-123", "Здравствуйте")

    assert 'data-chat-persist="true"' in html
    assert 'data-chat-role="assistant"' in html


def test_make_robot_blocks_html_renders_commentary_and_data_table_partials() -> None:
    configure_django_templates()

    html = make_robot_blocks_html(
        "robot-123",
        [
            {"id": "c1", "type": "commentary", "content": "**Готово**"},
            {
                "id": "t1",
                "type": "data_table",
                "title": "Клиенты",
                "columns": [{"key": "customer", "label": "Клиент", "type": "string"}],
                "rows": [{"customer": "ООО <Тест>"}],
                "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
            },
        ],
    )

    assert 'data-chat-blocks="' in html
    assert 'id="robot-123-block-c1"' in html
    assert 'data-block-type="commentary"' in html
    assert 'id="robot-123-block-t1"' in html
    assert 'data-block-type="data_table"' in html
    assert 'data-table-copy-button' in html
    assert 'aria-label="Скопировать таблицу"' in html
    assert "Клиенты" in html
    assert "Клиент" in html
    assert "ООО &lt;Тест&gt;" in html


def test_make_robot_blocks_html_renders_empty_data_table_state() -> None:
    configure_django_templates()

    html = make_robot_blocks_html(
        "robot-123",
        [
            {
                "id": "t1",
                "type": "data_table",
                "columns": [],
                "rows": [],
                "meta": {"row_count": 0, "rendered_row_count": 0, "truncated": False},
            }
        ],
    )

    assert "По этому запросу нет строк для отображения." in html
    assert "Показано строк: 0 из 0" in html
    assert 'data-table-copy-button' not in html


def test_make_robot_blocks_html_serializes_decimal_blocks_for_session_restore() -> None:
    configure_django_templates()

    html = make_robot_blocks_html(
        "robot-123",
        [
            {
                "id": "t1",
                "type": "data_table",
                "columns": [{"key": "weight_kg", "label": "weight_kg", "type": "number"}],
                "rows": [{"weight_kg": Decimal("1.25")}],
                "meta": {"row_count": 1, "rendered_row_count": 1, "truncated": False},
            }
        ],
    )

    assert 'data-chat-blocks="' in html
    assert "1.25" in html


def test_make_robot_blocks_html_hides_answer_details_without_permissions() -> None:
    configure_django_templates()

    html = make_robot_blocks_html("robot-123", [block_with_details()])

    assert "Детали" not in html
    assert "mssql_default" not in html
    assert "последние 30 дней" not in html
    assert "SELECT * FROM dbo.customer_orders" not in html
    persisted_blocks = extract_chat_blocks_json(html)
    assert "details" not in persisted_blocks
    assert "raw_sql" not in persisted_blocks


def test_make_robot_blocks_html_shows_notes_without_raw_sql_for_notes_permission() -> None:
    configure_django_templates()

    html = make_robot_blocks_html("robot-123", [block_with_details()], can_view_answer_notes=True)

    assert "Детали" in html
    assert "mssql_default" in html
    assert "dbo.customer_orders" in html
    assert "последние 30 дней" in html
    assert "model-chosen-source" not in html
    assert "SELECT * FROM dbo.customer_orders" not in html
    persisted_blocks = extract_chat_blocks_json(html)
    assert "Период" in persisted_blocks
    assert "model-chosen-source" not in persisted_blocks
    assert "raw_sql" not in persisted_blocks


def test_make_robot_blocks_html_shows_raw_sql_without_notes_for_sql_permission() -> None:
    configure_django_templates()

    html = make_robot_blocks_html("robot-123", [block_with_details()], can_view_raw_sql=True)

    assert "Детали" in html
    assert "Показать SQL" in html
    assert 'data-sql-copy-button="true"' in html
    assert "SELECT * FROM dbo.customer_orders" in html
    assert "последние 30 дней" not in html
    persisted_blocks = extract_chat_blocks_json(html)
    assert "raw_sql" in persisted_blocks
    assert "Период" not in persisted_blocks


def test_make_robot_blocks_html_shows_notes_and_raw_sql_with_both_permissions() -> None:
    configure_django_templates()

    html = make_robot_blocks_html(
        "robot-123",
        [block_with_details()],
        can_view_answer_notes=True,
        can_view_raw_sql=True,
    )

    assert "последние 30 дней" in html
    assert "SELECT * FROM dbo.customer_orders" in html
    persisted_blocks = extract_chat_blocks_json(html)
    assert "Период" in persisted_blocks
    assert "raw_sql" in persisted_blocks


def test_format_table_cell_value_formats_numbers_for_humans() -> None:
    assert format_table_cell_value(99_750) == "99 750"
    assert format_table_cell_value(Decimal("140697910.8300")) == "140 697 910,83"
    assert format_table_cell_value(Decimal("44622577.0000")) == "44 622 577"
    assert format_table_cell_value(True) == "True"
    assert format_table_cell_value(None) == ""
