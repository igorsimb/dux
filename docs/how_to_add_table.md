# How To Add New SQL Table

This is the single source of truth for adding a new table to the AI SQL catalog.

There are two paths:

1. New table in an existing source.
2. New source plus its first table.

## Before You Start

- Table discovery/allowlist lives in `ai/table_descriptions.json`.
- Detailed table card lives in `ai/table_metadata.json`.
- Source registry lives in `core/db_config/sql_sources.json`.
- Every table row in `ai/table_descriptions.json` must include `source` and `sql_dialect`.

## Required Config Shape

### `ai/table_descriptions.json` row

```json
{
  "table": "schema.table_name",
  "source": "clickhouse_default",
  "sql_dialect": "clickhouse",
  "summary": "Short practical table summary.",
  "tags": ["domain", "lookup"],
  "allowed": true
}
```

### `ai/table_metadata.json` card
For description of each field, see `docs/table_metadata_reference.md`.

```json
"schema.table_name": {
  "table": "schema.table_name",
  "description": "What this table stores and how it is used.",
  "grain": "Row per ...",
  "column_types": {
    "id": "Int64"
  },
  "important_columns": {
    "id": "business key"
  },
  "requires_date_filter": false,
  "sample_queries": [
    "SELECT id FROM schema.table_name LIMIT 10"
  ]
}
```

## Path A: New Table In Existing Source

Use this when source already exists in `core/db_config/sql_sources.json` (for example `clickhouse_default` or `mssql_default`).

1. Pick existing source id from `core/db_config/sql_sources.json`.
2. Add table row to `ai/table_descriptions.json`:
   - set `table`, `source`, `sql_dialect`, `summary`, `tags`, `allowed`.
3. Ensure `sql_dialect` matches source dialect from registry.
4. Add detailed card in `ai/table_metadata.json` under key equal to full table name.
5. Validate JSON and run focused tests.

Example (new MSSQL table in existing `mssql_default`):

```json
{
  "table": "dbo.sales_orders",
  "source": "mssql_default",
  "sql_dialect": "tsql",
  "summary": "ERP sales orders with amounts and statuses.",
  "tags": ["orders", "erp", "sales"],
  "allowed": true
}
```

## Path B: New Source + New Table

Use this when table is on a DB server not yet in source registry.

1. Add source entry to `core/db_config/sql_sources.json`.
2. Set `id`, `dialect`, `driver`, and required `env` mapping keys.
3. Set real env vars in your environment/secrets store.
4. Add table row to `ai/table_descriptions.json` using new source id.
5. Add matching table card to `ai/table_metadata.json`.
6. Validate JSON and run focused tests.

Example source entry (MS SQL):

```json
{
  "id": "mssql_finance",
  "dialect": "tsql",
  "driver": "mssql",
  "env": {
    "host": "MSSQL_FINANCE_HOST",
    "port": "MSSQL_FINANCE_PORT",
    "user": "MSSQL_FINANCE_USER",
    "password": "MSSQL_FINANCE_PASSWORD",
    "database": "MSSQL_FINANCE_DATABASE"
  }
}
```

Example table for that source:

```json
{
  "table": "dbo.payments",
  "source": "mssql_finance",
  "sql_dialect": "tsql",
  "summary": "Payments ledger with transaction status and timestamps.",
  "tags": ["finance", "payments", "ledger"],
  "allowed": true
}
```

## Validation Commands

```bash
python -m json.tool "ai/table_descriptions.json" > NUL
python -m json.tool "ai/table_metadata.json" > NUL
python -m json.tool "core/db_config/sql_sources.json" > NUL
python -m pytest tests/ai/test_sql_tools_config.py -q
python -m pytest tests/ai/test_ai_tools_validate_sql.py -q
```

## Common Mistakes

- Missing `source` or `sql_dialect` in table row.
- Unknown `source` id in `ai/table_descriptions.json`.
- `sql_dialect` mismatch between table row and source registry entry.
- Different table name between `table_descriptions` and `table_metadata`.
- Forgetting to set env vars for a newly added source.
