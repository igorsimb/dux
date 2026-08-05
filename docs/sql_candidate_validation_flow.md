# SQL Candidate Validation Flow

This document explains the NL->SQL pipeline split and the meaning of source+dialect candidates.

## Why We Split The Pipeline

The model gets table context before writing SQL, but output can still be ambiguous:

- unqualified table names (`customers`),
- syntax that parses in more than one dialect,
- references that do not map cleanly to one backend.

Because of this, generation and validation are separate phases:

1. Planning/generation (LLM proposes SQL).
2. Validation/routing (backend proves one deterministic execution route).

## What Is A Candidate

A candidate is one allowlisted routing option represented as `(source_id, sql_dialect)`.

Candidates are derived from allowlisted rows in `ai/table_descriptions.json`.

Example candidates:

- `("clickhouse_default", "clickhouse")`
- `("mssql_default", "tsql")`

## Validation Algorithm

For one SQL query, backend runs candidate checks in `validate_sql`:

1. Read-only guard.
2. Build allowlisted candidates.
3. For each candidate:
   - run dialect-specific pre-parse guards,
   - parse query with candidate dialect and resolve table names against that source allowlist,
   - canonicalize references to `db.table`,
   - enforce metadata policy (`requires_date_filter=true` -> predicate on `date_column`),
   - collect success or reject payload.
4. Resolve outcomes:
   - exactly one success -> issue `validated_id` with `source_id` and `dialect`,
   - zero successes -> reject with prioritized error,
   - multiple successes -> reject with `AMBIGUOUS_SOURCE_OR_DIALECT`.

## Why Prioritized Errors Exist

When all candidates fail, failures may differ by candidate. We select one stable primary error so retries are deterministic.

Priority:

1. `TABLE_NOT_ALLOWED`
2. `AMBIGUOUS_UNQUALIFIED_TABLE`
3. `MISSING_REQUIRED_DATE_FILTER`
4. `UNBOUND_QUERY_PARAMETER`
5. `SQL_PARSE_ERROR`

## End Result

Execution never uses raw SQL directly from model args.

- `validate_sql` issues a capability token (`validated_id`) only when routing is deterministic.
- `execute_validated_sql` executes only SQL stored under that token and uses stored routing metadata.
