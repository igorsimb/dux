import json
from pathlib import Path
import sqlite3

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


def test_checked_in_chinook_catalog_is_complete_and_consistent() -> None:
    project_root = Path(__file__).resolve().parents[2]
    descriptions = json.loads(
        (project_root / "ai" / "table_descriptions.json").read_text(encoding="utf-8")
    )["tables"]
    metadata = json.loads(
        (project_root / "ai" / "table_metadata.json").read_text(encoding="utf-8")
    )["tables"]
    expected_tables = {
        "Album",
        "Artist",
        "Customer",
        "Employee",
        "Genre",
        "Invoice",
        "InvoiceLine",
        "MediaType",
        "Playlist",
        "PlaylistTrack",
        "Track",
    }

    assert {row["table"] for row in descriptions} == expected_tables
    assert all(row["allowed"] is True for row in descriptions)
    assert all(row["source"] == "chinook" for row in descriptions)
    assert all(row["sql_dialect"] == "sqlite" for row in descriptions)
    assert set(metadata) == expected_tables

    for table_name, card in metadata.items():
        assert card["table"] == table_name
        assert card["description"]
        assert card["grain"]
        assert card["column_types"]
        assert card["important_columns"]
        assert card["query_hints"]
        assert card["sample_queries"]
        assert card["requires_date_filter"] is False


def test_checked_in_invoice_metadata_describes_historical_time_semantics() -> None:
    metadata_path = Path(__file__).resolve().parents[2] / "ai" / "table_metadata.json"
    invoice = json.loads(metadata_path.read_text(encoding="utf-8"))["tables"][
        "Invoice"
    ]
    hints = " ".join(invoice["query_hints"])

    assert "2009-01-01" in invoice["data_window"]
    assert "2013-12-22" in invoice["data_window"]
    assert "latest represented InvoiceDate" in hints
    assert "State this assumption" in hints


def test_bundled_chinook_database_matches_the_checked_in_catalog() -> None:
    project_root = Path(__file__).resolve().parents[2]
    descriptions = json.loads(
        (project_root / "ai" / "table_descriptions.json").read_text(encoding="utf-8")
    )["tables"]
    expected_tables = {row["table"] for row in descriptions}

    connection = sqlite3.connect(project_root / "Chinook.db")
    try:
        actual_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        date_bounds = connection.execute(
            "SELECT MIN(InvoiceDate), MAX(InvoiceDate) FROM Invoice"
        ).fetchone()
        top_genre = connection.execute(
            "SELECT g.Name, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS Revenue "
            "FROM InvoiceLine AS il "
            "JOIN Track AS t ON t.TrackId = il.TrackId "
            "JOIN Genre AS g ON g.GenreId = t.GenreId "
            "GROUP BY g.GenreId, g.Name ORDER BY Revenue DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()

    assert actual_tables == expected_tables
    assert date_bounds == ("2009-01-01 00:00:00", "2013-12-22 00:00:00")
    assert top_genre == ("Rock", 826.65)
