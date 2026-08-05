import json

from ai import ai_tools


def test_get_table_descriptions_includes_source_and_sql_dialect(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "load_table_descriptions",
        lambda: [
            {
                "table": "analytics.customers",
                "source": "clickhouse_default",
                "sql_dialect": "clickhouse",
                "summary": "Customer directory",
                "tags": ["customer"],
                "allowed": True,
            }
        ],
    )

    result = getattr(ai_tools.get_table_descriptions, "func")()

    payload = json.loads(result)
    assert len(payload) == 1
    assert payload[0]["table"] == "analytics.customers"
    assert payload[0]["source"] == "clickhouse_default"
    assert payload[0]["sql_dialect"] == "clickhouse"


def test_get_table_metadata_hides_disallowed_tables(monkeypatch) -> None:
    monkeypatch.setattr(ai_tools, "get_stream_writer", lambda: (lambda _text: None))
    monkeypatch.setattr(
        ai_tools,
        "load_table_metadata",
        lambda: {
            "analytics.customers": {
                "table": "analytics.customers",
                "description": "Customer directory",
            }
        },
    )

    result = getattr(ai_tools.get_table_metadata, "func")("analytics.hidden_table")

    assert result == "Table not found or not allowed."
