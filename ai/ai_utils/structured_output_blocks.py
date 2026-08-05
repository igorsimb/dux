"""Pydantic contracts for LangChain structured-output chat blocks."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_SQL_RESULT_ROW_LIMIT = 50
ColumnValueType = Literal["string", "number", "date", "datetime", "boolean", "unknown"]


class StrictBlockModel(BaseModel):
    """Base model for block contracts that should not accept undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class CommentaryBlock(StrictBlockModel):
    """Markdown commentary displayed around SQL result tables."""

    id: str
    type: Literal["commentary"] = "commentary"
    format: Literal["markdown"] = "markdown"
    content: str


class DataTablePlaceholderBlock(StrictBlockModel):
    """Positioned slot that the backend replaces with the next SQL result table."""

    id: str
    type: Literal["data_table_placeholder"] = "data_table_placeholder"
    title: str | None = None
    notes: list["AnswerDetailNote"] | None = None


class AnswerDetailNote(StrictBlockModel):
    """Model-supplied user-facing metadata note for one SQL result table."""

    label: str
    value: str


class TableColumn(StrictBlockModel):
    """Backend-produced metadata for one rendered data-table column."""

    key: str
    label: str
    type: ColumnValueType = "unknown"


class DataTableMeta(StrictBlockModel):
    """Backend-produced row-count metadata for a rendered SQL result table."""

    row_count: int
    rendered_row_count: int
    truncated: bool = False


class DataTableFacts(StrictBlockModel):
    """Backend-owned execution facts for one SQL result table."""

    source_id: str
    dialect: str
    validated_id: str
    tables: list[str]
    raw_sql: str


class DataTableDetails(StrictBlockModel):
    """Details attached to one backend-owned SQL result table."""

    facts: DataTableFacts | None = None
    notes: list[AnswerDetailNote] | None = None


class DataTableBlock(StrictBlockModel):
    """Backend-produced table block containing actual SQL result data.

    Only backend SQL execution may create this block. Its rows are the source of truth for the
    rendered table; the LLM may read the same rows through a ToolMessage, but it must not generate
    or rewrite `rows` for this block.
    """

    id: str
    type: Literal["data_table"] = "data_table"
    title: str | None = None
    columns: list[TableColumn]
    rows: list[dict[str, Any]]
    meta: DataTableMeta
    details: DataTableDetails | None = None


class AgentCommentaryResponse(StrictBlockModel):
    """Ordered commentary and SQL result-table placeholders."""

    blocks: list[CommentaryBlock | DataTablePlaceholderBlock] = Field(default_factory=list)


class AgentFinalResponse(StrictBlockModel):
    """Backend-assembled response rendered to the user.

    The backend builds this after resolving table placeholders against `sql_result_table_blocks`. This is
    the final UI-facing response contract and may contain backend-owned `DataTableBlock` objects.
    """

    blocks: list[CommentaryBlock | DataTableBlock] = Field(default_factory=list)


def build_data_table_block(
    *,
    block_id: str,
    rows: list[dict[str, Any]],
    title: str | None = None,
    row_limit: int = DEFAULT_SQL_RESULT_ROW_LIMIT,
) -> DataTableBlock:
    """Build a backend-owned table block from SQL row dictionaries."""

    rendered_rows = rows[:row_limit]
    column_keys = collect_column_keys(rendered_rows or rows)
    normalized_rows = [normalize_row(row, column_keys) for row in rendered_rows]
    return DataTableBlock(
        id=block_id,
        title=title,
        columns=[
            TableColumn(key=key, label=key, type=infer_column_type(rows, key)) for key in column_keys
        ],
        rows=normalized_rows,
        meta=DataTableMeta(
            row_count=len(rows),
            rendered_row_count=len(normalized_rows),
            truncated=len(rows) > len(normalized_rows),
        ),
    )


def collect_column_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Return column keys in first-seen row order."""

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def normalize_row(row: dict[str, Any], column_keys: list[str]) -> dict[str, Any]:
    """Return a row containing every rendered column key."""

    return {key: row.get(key) for key in column_keys}


def infer_column_type(rows: list[dict[str, Any]], key: str) -> ColumnValueType:
    """Infer one table column type from concrete Python values."""

    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return "unknown"
    inferred_types = {infer_value_type(value) for value in values}
    if len(inferred_types) != 1:
        return "unknown"
    return inferred_types.pop()


def infer_value_type(value: Any) -> ColumnValueType:
    """Infer the table column type for one Python value."""

    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, int | float | Decimal):
        return "number"
    if isinstance(value, str):
        return "string"
    return "unknown"
