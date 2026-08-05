import pytest
from pydantic import ValidationError

from ai.ai_utils.structured_output_blocks import (
    AgentCommentaryResponse,
    AgentFinalResponse,
    AnswerDetailNote,
    CommentaryBlock,
    DataTableBlock,
    DataTableDetails,
    DataTableFacts,
    DataTableMeta,
    DataTablePlaceholderBlock,
    TableColumn,
)


def test_commentary_block_validates_and_serializes() -> None:
    block = CommentaryBlock(id="c1", content="Вот **результаты**:")

    assert block.type == "commentary"
    assert block.format == "markdown"
    assert block.model_dump() == {
        "id": "c1",
        "type": "commentary",
        "format": "markdown",
        "content": "Вот **результаты**:",
    }


def test_data_table_placeholder_validates_without_rows() -> None:
    block = DataTablePlaceholderBlock(id="p1", title="Топ клиентов")

    assert block.type == "data_table_placeholder"
    assert block.model_dump() == {
        "id": "p1",
        "type": "data_table_placeholder",
        "title": "Топ клиентов",
        "notes": None,
    }


def test_data_table_placeholder_accepts_labeled_notes() -> None:
    block = DataTablePlaceholderBlock(
        id="p1",
        title="Топ клиентов",
        notes=[AnswerDetailNote(label="Период", value="последние 30 дней")],
    )

    assert block.notes is not None
    assert block.notes[0].label == "Период"
    assert block.model_dump()["notes"] == [{"label": "Период", "value": "последние 30 дней"}]


def test_data_table_block_validates_and_serializes() -> None:
    block = DataTableBlock(
        id="t1",
        title="Результаты запроса",
        columns=[
            TableColumn(key="customer", label="Клиент", type="string"),
            TableColumn(key="qty", label="Количество", type="number"),
        ],
        rows=[
            {"customer": "A", "qty": 10},
            {"customer": "B", "qty": 20},
        ],
        meta=DataTableMeta(row_count=200, rendered_row_count=2, truncated=True),
        details=DataTableDetails(
            facts=DataTableFacts(
                source_id="mssql_default",
                dialect="tsql",
                validated_id="vid-1",
                tables=["dbo.customer_orders"],
                raw_sql="SELECT * FROM dbo.customer_orders",
            ),
            notes=[AnswerDetailNote(label="Метрика", value="количество продаж")],
        ),
    )

    assert block.type == "data_table"
    assert block.meta.row_count == 200
    assert block.meta.rendered_row_count == 2
    assert block.meta.truncated is True
    assert block.model_dump()["rows"] == [
        {"customer": "A", "qty": 10},
        {"customer": "B", "qty": 20},
    ]
    details = block.model_dump()["details"]
    assert details["facts"]["source_id"] == "mssql_default"
    assert details["facts"]["tables"] == ["dbo.customer_orders"]
    assert details["notes"] == [{"label": "Метрика", "value": "количество продаж"}]


def test_table_column_type_defaults_to_unknown() -> None:
    column = TableColumn(key="mixed", label="Mixed")

    assert column.type == "unknown"


def test_data_table_rows_accept_common_json_values() -> None:
    block = DataTableBlock(
        id="t1",
        columns=[TableColumn(key="value", label="Value")],
        rows=[
            {"value": "text"},
            {"value": 10},
            {"value": 10.5},
            {"value": True},
            {"value": None},
        ],
        meta=DataTableMeta(row_count=5, rendered_row_count=5),
    )

    assert [row["value"] for row in block.rows] == ["text", 10, 10.5, True, None]


def test_agent_commentary_response_accepts_commentary_and_placeholder_blocks() -> None:
    response = AgentCommentaryResponse.model_validate(
        {
            "blocks": [
                {"id": "c1", "type": "commentary", "content": "Вот результаты:"},
                {"id": "p1", "type": "data_table_placeholder", "title": "Таблица"},
            ]
        }
    )

    assert isinstance(response.blocks[0], CommentaryBlock)
    assert isinstance(response.blocks[1], DataTablePlaceholderBlock)


def test_agent_commentary_response_rejects_data_table_blocks() -> None:
    with pytest.raises(ValidationError):
        AgentCommentaryResponse.model_validate(
            {
                "blocks": [
                    {
                        "id": "t1",
                        "type": "data_table",
                        "columns": [],
                        "rows": [],
                        "meta": {"row_count": 0, "rendered_row_count": 0},
                    }
                ]
            }
        )


def test_agent_final_response_accepts_commentary_and_data_table_blocks() -> None:
    response = AgentFinalResponse.model_validate(
        {
            "blocks": [
                {"id": "c1", "type": "commentary", "content": "Вот результаты:"},
                {
                    "id": "t1",
                    "type": "data_table",
                    "columns": [{"key": "customer", "label": "Клиент"}],
                    "rows": [{"customer": "A"}],
                    "meta": {"row_count": 1, "rendered_row_count": 1},
                },
            ]
        }
    )

    assert isinstance(response.blocks[0], CommentaryBlock)
    assert isinstance(response.blocks[1], DataTableBlock)


def test_agent_final_response_rejects_placeholder_blocks() -> None:
    with pytest.raises(ValidationError):
        AgentFinalResponse.model_validate(
            {"blocks": [{"id": "p1", "type": "data_table_placeholder"}]}
        )
