# Table metadata reference

## Why metadata exists

An allowlist answers which tables the model may query. Metadata explains how to query those tables correctly without
making the model the authority for SQL safety. Dux keeps these responsibilities separate:

- `ai/table_descriptions.json` defines model-visible tables, source routing, dialect, and allowlist status.
- `ai/table_metadata.json` describes columns, table grain, date-filter requirements, and query guidance.
- `ai/sql_guard.py` and `ai/ai_utils/validate_sql.py` enforce deterministic policy.

The public repository ships with a populated catalog for the bundled Chinook sample database. The examples below remain
fictional so they can demonstrate the file format independently of that example catalog.

## Table descriptions

The root object must contain a `tables` list. Each row represents one table:

```json
{
  "tables": [
    {
      "table": "analytics.sales_fact",
      "source": "clickhouse_default",
      "sql_dialect": "clickhouse",
      "allowed": true,
      "summary": "Fictional daily sales events",
      "tags": ["sales", "orders"]
    }
  ]
}
```

Required routing fields:

- `table`: exact canonical table name; normally `schema.table`, or an unqualified table such as `Invoice` for SQLite
- `source`: source ID from `core/db_config/sql_sources.json`
- `sql_dialect`: dialect matching that source
- `allowed`: whether the table is exposed through the catalog

Additional fields such as `summary` and `tags` are model-facing context. They do not override validation policy.

## Detailed table metadata

The root object must contain a `tables` object keyed by the same canonical names used in the description file:

```json
{
  "tables": {
    "analytics.sales_fact": {
      "table": "analytics.sales_fact",
      "description": "Fictional daily sales events for documentation and tests.",
      "grain": "One row per order line.",
      "requires_date_filter": true,
      "date_column": "event_date",
      "important_columns": {
        "event_date": "Business date of the sale",
        "order_id": "Order identifier",
        "customer_id": "Customer identifier",
        "product_id": "Product identifier",
        "quantity": "Units sold",
        "revenue": "Line revenue in the source currency"
      },
      "query_hints": [
        "Filter event_date to a concrete period.",
        "Use SUM(revenue) for total revenue and SUM(quantity) for units sold."
      ],
      "sample_queries": [
        "SELECT event_date, SUM(revenue) AS revenue FROM analytics.sales_fact WHERE event_date >= today() - 30 GROUP BY event_date ORDER BY event_date"
      ]
    }
  }
}
```

## Field semantics

### `table`

Use the exact canonical table name. It must match a corresponding allowed row in
`ai/table_descriptions.json`. Metadata for tables outside the allowlist is filtered out before it reaches the model.

### `description`

Describe what the table represents and when it is useful. Avoid copying internal runbooks, credentials, ownership
details, hostnames, or operational incident notes.

### `grain`

State what one row represents. Grain helps prevent accidental double counting and unsafe joins.

### `requires_date_filter` and `date_column`

Set `requires_date_filter` to `true` for large fact tables that must be bounded by time. When enabled, `date_column`
names the column that deterministic validation requires in a filter expression.

### `important_columns`

Map column names to concise business meanings. Include only columns needed for safe, useful queries. Do not expose
private identifiers or sensitive fields unless the application is explicitly authorized to query them.

### `query_hints`

Use hints for semantic guidance such as preferred measures, valid join keys, null handling, or known aggregation rules.
Hints inform SQL generation but do not grant access or bypass deterministic checks.

### `sample_queries`

Examples should be read-only, use exact allowlisted table names with qualification where the source requires it, match
the source dialect, and contain fictional or public sample values. Never copy production queries into a public catalog.

## A fictional MSSQL example

```json
{
  "table": "dbo.customer_orders",
  "description": "Fictional order headers used in documentation examples.",
  "grain": "One row per customer order.",
  "requires_date_filter": false,
  "important_columns": {
    "order_id": "Order identifier",
    "customer_id": "Customer identifier",
    "ordered_at": "Order timestamp",
    "total_amount": "Total order amount"
  },
  "sample_queries": [
    "SELECT TOP (20) order_id, customer_id, ordered_at, total_amount FROM dbo.customer_orders ORDER BY total_amount DESC"
  ]
}
```

## Validation behavior

Configuration loaders fail fast when:

- JSON is malformed or has the wrong root shape.
- A table row omits its source or dialect.
- A source ID is unknown.
- A table dialect conflicts with its source dialect.
- Metadata has the wrong `tables` shape.

Run the focused catalog and routing tests after every catalog change. Static configuration validation should not connect
to a live database.

## Where to look in the code

- `ai/ai_utils/sql_tools.py`: catalog loading, normalization, and filtering
- `ai/ai_utils/validate_sql.py`: required-date-filter and candidate validation
- `ai/sql_guard.py`: read-only and allowlist enforcement
- `core/db_config/source_registry.py`: source configuration validation
- `core/db_config/source_database_router.py`: source-specific table routing
- `tests/ai/test_sql_tools_config.py`: deterministic catalog tests
