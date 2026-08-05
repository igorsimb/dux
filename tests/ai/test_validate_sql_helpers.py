import sqlglot

from ai.ai_utils import validate_sql as validate_sql_helpers
from ai.sql_dialect import SQLDialect


def _parse_clickhouse(query: str):
    return sqlglot.parse_one(query, read=SQLDialect.CLICKHOUSE)


def test_check_required_date_filter_rejects_when_missing_predicate(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers,
        "load_table_metadata",
        lambda: {
            "analytics.sales_fact": {
                "requires_date_filter": True,
                "date_column": "event_date",
            }
        },
    )
    parsed_ast = _parse_clickhouse(
        "SELECT region, product_id, revenue FROM analytics.sales_fact WHERE region = 'North'"
    )

    result = validate_sql_helpers.check_required_date_filter_for_tables(
        parsed_ast, ["analytics.sales_fact"]
    )

    assert isinstance(result, dict)
    assert result.get("error_code") == "MISSING_REQUIRED_DATE_FILTER"
    assert result.get("tables_missing_date_filter") == [
        "analytics.sales_fact"
    ]


def test_check_required_date_filter_passes_when_predicate_present(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers,
        "load_table_metadata",
        lambda: {
            "analytics.sales_fact": {
                "requires_date_filter": True,
                "date_column": "event_date",
            }
        },
    )
    parsed_ast = _parse_clickhouse(
        "SELECT region, product_id, revenue FROM analytics.sales_fact "
        "WHERE event_date >= today() - 7 AND region = 'North'"
    )

    result = validate_sql_helpers.check_required_date_filter_for_tables(
        parsed_ast, ["analytics.sales_fact"]
    )

    assert result is None


def test_check_required_date_filter_works_with_table_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers,
        "load_table_metadata",
        lambda: {
            "analytics.sales_fact": {
                "requires_date_filter": True,
                "date_column": "event_date",
            }
        },
    )
    parsed_ast = _parse_clickhouse(
        "SELECT sf.region FROM analytics.sales_fact AS sf WHERE sf.event_date >= today() - 3"
    )

    result = validate_sql_helpers.check_required_date_filter_for_tables(
        parsed_ast, ["analytics.sales_fact"]
    )

    assert result is None


def test_check_required_date_filter_works_with_cte_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_sql_helpers,
        "load_table_metadata",
        lambda: {
            "analytics.sales_fact": {
                "requires_date_filter": True,
                "date_column": "event_date",
            }
        },
    )
    parsed_ast = _parse_clickhouse(
        "WITH recent_sales AS (SELECT * FROM analytics.sales_fact) "
        "SELECT * FROM recent_sales WHERE recent_sales.event_date >= today() - 7"
    )

    result = validate_sql_helpers.check_required_date_filter_for_tables(
        parsed_ast, ["analytics.sales_fact"]
    )

    assert result is None


def test_check_required_date_filter_skips_tables_without_requirement(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        validate_sql_helpers,
        "load_table_metadata",
        lambda: {
            "analytics.customers": {
                "requires_date_filter": False,
            }
        },
    )
    parsed_ast = _parse_clickhouse("SELECT customer_id, name FROM analytics.customers")

    result = validate_sql_helpers.check_required_date_filter_for_tables(
        parsed_ast, ["analytics.customers"]
    )

    assert result is None


def test_select_prioritized_candidate_error_prefers_date_filter_over_parse() -> None:
    errors = [
        {
            "type": "reject",
            "error_code": "SQL_PARSE_ERROR",
            "reason": "bad syntax",
        },
        {
            "type": "reject",
            "error_code": "MISSING_REQUIRED_DATE_FILTER",
            "reason": "missing date",
        },
    ]

    selected = validate_sql_helpers.select_prioritized_candidate_error(errors)

    assert isinstance(selected, dict)
    assert selected.get("error_code") == "MISSING_REQUIRED_DATE_FILTER"


def test_narrow_candidates_to_explicit_table_sources_uses_allowlisted_full_names() -> None:
    candidates = [
        ("clickhouse_default", SQLDialect.CLICKHOUSE),
        ("mssql_default", SQLDialect.MSSQL),
    ]

    narrowed, allowed_tables_by_source, referenced_tables = (
        validate_sql_helpers.narrow_candidates_to_explicit_table_sources(
            query="SELECT * FROM analytics.sales_fact",
            candidates=candidates,
            load_allowed_tables_for_source=lambda source_id: {
                "clickhouse_default": {"analytics.sales_fact", "analytics.customers"},
                "mssql_default": {"dbo.customer_orders"},
            }[source_id],
        )
    )

    assert narrowed == [("clickhouse_default", SQLDialect.CLICKHOUSE)]
    assert allowed_tables_by_source["clickhouse_default"] == {
        "analytics.sales_fact",
        "analytics.customers",
    }
    assert referenced_tables == ["analytics.sales_fact"]
