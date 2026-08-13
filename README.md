# Dux

Dux is a slightly sarcastic guarded natural-language-to-SQL application for querying business data. It combines a 
Django chat interface
with LangChain and LangGraph to inspect an allowlisted catalog, generate SQL, validate it deterministically, execute it
against SQLite, ClickHouse, or Microsoft SQL Server, and render backend-owned structured results.

The public repository includes the Chinook sample music-store database and a complete checked-in catalog, so a fresh
clone can demonstrate the guarded query flow without an external database. Deployments can replace or extend the
example with their own sources, allowlisted tables, and model-facing metadata.

## Why the SQL flow is guarded

Model-generated SQL is never executed directly. Every query crosses a two-step capability boundary:

1. `validate_sql(query)` checks read-only safety, SQL dialect, table qualification, source routing, and the table
   allowlist. A successful validation returns a short-lived, thread-bound, one-time `validated_id`.
2. `execute_validated_sql(validated_id)` retrieves the backend-owned validated query and executes it once.

A query must validate successfully for exactly one configured source and dialect. Queries that are unsafe, ambiguous,
cross-source, or outside the allowlist are rejected before execution.

## Main features

- Authenticated English-first chat UI built with Django, Datastar, and SSE
- One LangGraph conversation with deterministic English/Russian routing for SQL, small talk, and theme changes
- SQLite, ClickHouse, and Microsoft SQL Server source support
- Deterministic read-only SQL validation with `sqlglot`
- Per-source table allowlists and detailed metadata
- Backend-owned result rows and structured UI blocks
- Permission-gated raw SQL and model notes
- Per-tab short-term chat continuity and token-usage reporting
- Application and chat transcript logging with sensitive detail filtering

## Project layout

- `ai/views.py`: async HTTP and SSE request lifecycle
- `ai/ai_utils/chat_runtime.py`: runtime entry point and source-specific SQL tool construction
- `ai/ai_utils/chat_agent.py`: shared LangGraph agent and middleware stack
- `ai/ai_utils/intent_router.py`: deterministic per-turn intent routing
- `ai/ai_tools.py`: model-visible catalog, validation, and validated execution tools
- `ai/ai_utils/validate_sql.py`: candidate-based multi-source validation
- `ai/sql_guard.py`: deterministic SQL safety and allowlist enforcement
- `ai/ai_utils/streaming.py`: graph-event to SSE translation
- `ai/ai_utils/structured_output_blocks.py`: backend-owned result blocks
- `core/db_config/`: source registry and database connector routing
- `tests/`: focused deterministic test suite

## Configure data sources

Sources are declared in `core/db_config/sql_sources.json`. The public configuration includes the bundled SQLite
`chinook` example plus neutral ClickHouse and MSSQL source definitions whose connection values are resolved from
environment variables.

The catalog is split across two files:

- `ai/table_descriptions.json` lists model-visible allowlisted tables and their source and dialect.
- `ai/table_metadata.json` describes columns, query constraints, and safe example queries for each allowlisted table.

Both files contain the 11-table Chinook catalog by default. See `docs/how_to_add_table.md` and
`docs/table_metadata_reference.md` before replacing or adding deployment-specific catalog entries. Treat catalog
changes as security changes because they determine what SQL the model can validate and execute.

## Run locally

Requirements:

- Python 3.13
- ODBC Driver 18 for SQL Server when using MSSQL
- Access to any additional configured external data source

Install dependencies and apply Django migrations:

```powershell
uv sync
.venv/Scripts/python.exe manage.py migrate
```

Start the supported ASGI server:

```powershell
.venv/Scripts/python.exe -m daphne config.asgi:application
```

Open the local address printed by Daphne and sign in with an account provisioned by an administrator.

## Environment variables

Create a local `.env` file. It is ignored by Git and excluded from Docker build context.

Application and model settings:

- `DJANGO_SECRET_KEY`: required Django signing key
- `DJANGO_DEBUG`: optional, defaults to `1`
- `DJANGO_ALLOWED_HOSTS`: optional comma-separated host list, defaults to `*`
- `SQLITE_PATH`: optional application database path
- `OPENAI_API_KEY`: required for model calls
- `OPENAI_MODEL`: optional model selection
- `OPENAI_PROXY`: optional proxy URL
- `CHINOOK_DATABASE_PATH`: optional Chinook data-source path; defaults to repository-root `Chinook.db`

Default ClickHouse source:

- `CLICKHOUSE_HOST`
- `CLICKHOUSE_PORT`
- `CLICKHOUSE_USER`
- `CLICKHOUSE_PASSWORD`

Default MSSQL source:

- `MSSQL_HOST`
- `MSSQL_PORT`
- `MSSQL_USER`
- `MSSQL_PASSWORD`
- `MSSQL_DATABASE`
- `MSSQL_ODBC_DRIVER`: optional, defaults to `ODBC Driver 18 for SQL Server`

Never commit a populated `.env`, connection values, production catalog metadata, database dumps, application logs, or
chat transcripts.

## Tests

Most tests mock model and database boundaries and require no API key or live database.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe manage.py check
```

## Documentation

- `docs/intent-router.md`: intent routing and middleware behavior
- `docs/sql_candidate_validation_flow.md`: multi-source validation flow
- `docs/structured-output-blocks.md`: backend-owned result rendering
- `docs/progress-messages.md`: user-facing progress updates
- `docs/app-logging.md`: application and chat logging
- `docs/how_to_add_table.md`: catalog and source configuration workflow

## License

See `LICENSE`.
