"""Datastar UI helpers for chat message HTML formatting and SSE element patches."""

from __future__ import annotations

import json
import re
from decimal import Decimal

from datastar_py.consts import ElementPatchMode
from datastar_py.django import ServerSentEventGenerator as SSE
from django.template.loader import render_to_string
from django.utils.html import escape

CHAT_MESSAGES_SELECTOR = "#chat-messages"
ROBOT_ID_PREFIX = "robot-"
RESERVED_ANSWER_DETAIL_NOTE_LABELS = {"sql", "source", "rows shown", "tables"}


def format_chat_html(text: str) -> str:
    """Escape text for HTML and convert newlines to <br>."""
    normalized = re.sub(r"(?<!\n)```", "\n```", text)
    return escape(normalized).replace("\n", "<br>")


def make_user_message_html(text: str) -> str:
    """Build HTML for a user message bubble."""
    safe_text = format_chat_html(text)
    return (
        '<div class="chat chat-end" data-chat-persist="true" data-chat-role="user">'
        f'<div class="chat-bubble chat-bubble-primary">{safe_text}</div>'
        "</div>"
    )


def make_robot_container_html(robot_id: str) -> str:
    """Build an empty container for the robot response."""
    return f'<div id="{robot_id}" class="chat chat-start"></div>'


def make_robot_message_html(robot_id: str, text: str) -> str:
    """Build HTML for the robot response bubble."""
    safe_text = format_chat_html(text)
    return (
        f'<div id="{robot_id}" class="chat chat-start" data-chat-persist="true" data-chat-role="assistant">'
        '<div class="chat-header text-xs opacity-70 mb-1">Dux</div>'
        f'<div class="chat-bubble">{safe_text}</div>'
        "</div>"
    )


def make_robot_blocks_html(
    robot_id: str,
    blocks: list[dict],
    *,
    can_view_answer_notes: bool = False,
    can_view_raw_sql: bool = False,
) -> str:
    """Build HTML for a structured robot response using typed block templates."""

    visible_blocks = project_blocks_for_answer_details_permissions(
        blocks,
        can_view_answer_notes=can_view_answer_notes,
        can_view_raw_sql=can_view_raw_sql,
    )
    rendered_blocks = "".join(format_structured_block_html(robot_id, block) for block in visible_blocks)
    blocks_json = escape(json.dumps(visible_blocks, ensure_ascii=False, default=str))
    return (
        f'<div id="{escape(robot_id)}" class="chat chat-start" data-chat-persist="true" '
        f'data-chat-role="assistant" data-chat-blocks="{blocks_json}">'
        '<div class="chat-header text-xs opacity-70 mb-1">Dux</div>'
        f'<div class="chat-bubble max-w-full space-y-3">{rendered_blocks}</div>'
        "</div>"
    )


def project_blocks_for_answer_details_permissions(
    blocks: list[dict],
    *,
    can_view_answer_notes: bool = False,
    can_view_raw_sql: bool = False,
) -> list[dict]:
    """Return blocks with restricted answer details removed before rendering or persistence."""

    return [
        project_block_for_answer_details_permissions(
            block,
            can_view_answer_notes=can_view_answer_notes,
            can_view_raw_sql=can_view_raw_sql,
        )
        for block in blocks
    ]


def project_block_for_answer_details_permissions(
    block: dict,
    *,
    can_view_answer_notes: bool = False,
    can_view_raw_sql: bool = False,
) -> dict:
    if block.get("type") != "data_table":
        return block

    details = block.get("details") if isinstance(block.get("details"), dict) else None
    if not details:
        return block

    facts = details.get("facts") if isinstance(details.get("facts"), dict) else None
    notes = details.get("notes") if isinstance(details.get("notes"), list) else None
    projected_notes = filter_model_detail_notes(notes or [])
    projected_details: dict[str, object] = {}
    if can_view_answer_notes and projected_notes:
        projected_details["notes"] = projected_notes
    if facts and (can_view_answer_notes or can_view_raw_sql):
        projected_facts = {key: value for key, value in facts.items() if key != "raw_sql"}
        if can_view_raw_sql and facts.get("raw_sql"):
            projected_facts["raw_sql"] = facts["raw_sql"]
        if projected_facts:
            projected_details["facts"] = projected_facts
    if projected_details:
        return {**block, "details": projected_details}
    return {key: value for key, value in block.items() if key != "details"}


def filter_model_detail_notes(notes: list[object]) -> list[dict[str, object]]:
    """Drop model notes that collide with backend-owned detail labels."""

    filtered_notes: list[dict[str, object]] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        if is_reserved_answer_detail_note_label(note.get("label")):
            continue
        filtered_notes.append(note)
    return filtered_notes


def is_reserved_answer_detail_note_label(label: object) -> bool:
    normalized_label = str(label or "").strip().casefold().removesuffix(":").strip()
    return normalized_label in RESERVED_ANSWER_DETAIL_NOTE_LABELS


def format_structured_block_html(robot_id: str, block: dict) -> str:
    """Render one structured block through its dedicated template."""

    block_type = str(block.get("type") or "")
    block_id = str(block.get("id") or "")
    block_dom_id = f"{robot_id}-block-{block_id}"
    if block_type == "commentary":
        return render_to_string(
            "ai/partials/commentary_block.html#commentary-block",
            {"block": block, "block_dom_id": block_dom_id},
        )
    if block_type == "data_table":
        return render_to_string(
            "ai/partials/data_table_block.html#data-table-block",
            {"block": prepare_data_table_block_for_template(block), "block_dom_id": block_dom_id},
        )
    return ""


def prepare_data_table_block_for_template(block: dict) -> dict:
    """Return table rows ordered by column metadata for template rendering."""

    columns = block.get("columns") if isinstance(block.get("columns"), list) else []
    rows = block.get("rows") if isinstance(block.get("rows"), list) else []
    column_keys = [str(column.get("key") or "") for column in columns if isinstance(column, dict)]
    rendered_rows = []
    for row in rows:
        source = row if isinstance(row, dict) else {}
        rendered_rows.append([format_table_cell_value(source.get(key)) for key in column_keys])
    return {
        **block,
        "columns": columns,
        "rendered_rows": rendered_rows,
        "details": prepare_table_details(block),
    }


def prepare_table_details(block: dict) -> dict:
    details = block.get("details") if isinstance(block.get("details"), dict) else {}
    facts = details.get("facts") if isinstance(details.get("facts"), dict) else {}
    notes = details.get("notes") if isinstance(details.get("notes"), list) else []
    meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
    rendered_row_count = int(meta.get("rendered_row_count") or 0)
    row_count = int(meta.get("row_count") or rendered_row_count)
    source_id = str(facts.get("source_id") or "").strip()
    tables = [str(table) for table in facts.get("tables") or [] if str(table).strip()]
    raw_sql = str(facts.get("raw_sql") or "").strip()
    prepared_notes = [
        {
            "label": str(note.get("label") or "").strip(),
            "value": str(note.get("value") or "").strip(),
        }
        for note in notes
        if isinstance(note, dict)
        and str(note.get("label") or "").strip()
        and str(note.get("value") or "").strip()
        and not is_reserved_answer_detail_note_label(note.get("label"))
    ]
    return {
        "has_details": bool(source_id or tables or prepared_notes or raw_sql),
        "source_id": source_id,
        "tables": tables,
        "rendered_row_count": rendered_row_count,
        "row_count": row_count,
        "notes": prepared_notes,
        "raw_sql": raw_sql,
    }


def format_table_cell_value(value) -> str:
    """Convert SQL cell values to display text for HTML templates.

    Examples:
        >>> format_table_cell_value(99750)
        '99 750'
        >>> format_table_cell_value(Decimal("140697910.8300"))
        '140 697 910,83'
        >>> format_table_cell_value(Decimal("44622577.0000"))
        '44 622 577'
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    if isinstance(value, Decimal):
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return f"{int(normalized):,}".replace(",", " ")
        return f"{value:,.2f}".replace(",", " ").replace(".", ",")
    return str(value)


def build_blocks_visible_text(blocks: list[dict]) -> str:
    """Return plain visible transcript text for structured blocks."""

    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "commentary":
            content = str(block.get("content") or "").strip()
            if content:
                parts.append(content)
        elif block.get("type") == "data_table":
            title = str(block.get("title") or "Query results")
            meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
            rendered_row_count = int(meta.get("rendered_row_count") or 0)
            row_count = int(meta.get("row_count") or rendered_row_count)
            parts.append(
                f"[Table: {title}, rows shown: {rendered_row_count} of {row_count}]"
            )
    return "\n\n".join(parts)


def append_user_message(text: str):
    """Create an SSE patch for a user message bubble."""
    return SSE.patch_elements(
        make_user_message_html(text),
        mode=ElementPatchMode.APPEND,
        selector=CHAT_MESSAGES_SELECTOR,
    )


def append_robot_container(robot_id: str):
    """Create an SSE patch for an empty robot container."""
    return SSE.patch_elements(
        make_robot_container_html(robot_id),
        mode=ElementPatchMode.APPEND,
        selector=CHAT_MESSAGES_SELECTOR,
    )


def append_robot_text(robot_id: str, text: str):
    """Create an SSE patch for the robot response text."""
    return SSE.patch_elements(make_robot_message_html(robot_id, text))


def append_robot_blocks(
    robot_id: str,
    blocks: list[dict],
    *,
    can_view_answer_notes: bool = False,
    can_view_raw_sql: bool = False,
):
    """Create an SSE patch for a structured robot response."""
    return SSE.patch_elements(
        make_robot_blocks_html(
            robot_id,
            blocks,
            can_view_answer_notes=can_view_answer_notes,
            can_view_raw_sql=can_view_raw_sql,
        )
    )
