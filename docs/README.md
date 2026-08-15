# Dux documentation

Dux is a guarded natural-language-to-SQL application. The model can inspect a catalog and compose the query a user
needs, while deterministic application code decides whether that query is safe, unambiguous, and allowed to run.

If you are new to the project, start with the repository [README](../README.md) for the product overview and local
Quickstart. This page is the entry point for architecture, configuration, and development details.

## Understand the system

- [SQL candidate validation flow](sql_candidate_validation_flow.md) explains how one query is checked against every
  configured source and dialect, and why exactly one candidate must succeed.
- [Table metadata reference](table_metadata_reference.md) describes the allowlist and the model-facing catalog used to
  generate accurate SQL.
- [Structured output blocks](structured-output-blocks.md) explains why result rows remain backend-owned instead of being
  reconstructed by the model.
- [Intent routing](intent-router.md) covers deterministic routing for SQL questions, small talk, and theme changes while
  preserving one conversation.
- [HITL clarification flow](hitl-clarification-flow.md) describes how ambiguous data questions pause for user input and
  resume safely.

## Configure Dux

- [How to add a SQL table](how_to_add_table.md) is the end-to-end workflow for extending an existing source or adding a
  new one.
- [Table metadata reference](table_metadata_reference.md) documents every supported catalog field and its effect on
  model context and validation.

Sources are declared in `core/db_config/sql_sources.json`. The public configuration contains:

- `chinook`, a bundled SQLite source that works immediately after setup;
- `clickhouse_default`, a neutral ClickHouse source configured through environment variables;
- `mssql_default`, a neutral Microsoft SQL Server source configured through environment variables.

The model-visible catalog is split across two files:

- `ai/table_descriptions.json` declares tables, sources, SQL dialects, summaries, and allowlist status.
- `ai/table_metadata.json` describes columns, relationships, constraints, and concise example queries.

Treat changes to either file as security-sensitive configuration. Together they define what the model can discover and
what generated SQL can pass deterministic validation.

## Environment variables

Copy `.env.example` to `.env` for local development. The populated `.env` file is ignored by Git and excluded from the
Docker build context.

### Application and model

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Authenticates model calls. |
| `OPENAI_MODEL` | No | Overrides the model selected by `ai/ai_utils/runtime_config.py`. |
| `OPENAI_PROXY` | No | Routes OpenAI traffic through an optional proxy. |
| `DJANGO_SECRET_KEY` | Deployment | Replaces the checked-in development-only secret. |
| `DJANGO_DEBUG` | No | Enables Django debug mode when set to `1`; defaults to `1` locally. |
| `DJANGO_ALLOWED_HOSTS` | No | Comma-separated allowed hosts; defaults to `*`. |
| `SQLITE_PATH` | No | Overrides the SQLite file used by the Django application itself. |
| `CHINOOK_DATABASE_PATH` | No | Overrides the bundled repository-root `Chinook.db` data source. |

### ClickHouse source

| Variable | Required when used | Purpose |
| --- | --- | --- |
| `CLICKHOUSE_HOST` | Yes | ClickHouse host. |
| `CLICKHOUSE_PORT` | Yes | ClickHouse HTTP port; the example uses `8123`. |
| `CLICKHOUSE_USER` | Yes | ClickHouse user. |
| `CLICKHOUSE_PASSWORD` | Yes | ClickHouse password. |

### Microsoft SQL Server source

| Variable | Required when used | Purpose |
| --- | --- | --- |
| `MSSQL_HOST` | Yes | SQL Server host. |
| `MSSQL_PORT` | Yes | SQL Server port; the example uses `1433`. |
| `MSSQL_USER` | Yes | SQL Server user. |
| `MSSQL_PASSWORD` | Yes | SQL Server password. |
| `MSSQL_DATABASE` | Yes | Database name. |
| `MSSQL_ODBC_DRIVER` | No | ODBC driver name; defaults to `ODBC Driver 18 for SQL Server`. |

Never commit production secrets, connection values, production catalog metadata, database dumps, application logs, or
chat transcripts.

## Project layout

- `config/settings.py` contains Django settings, the SQLite application database, authentication, and environment
  loading.
- `ai/views.py` owns the async HTTP and SSE request lifecycle.
- `ai/ai_utils/chat_runtime.py` builds the model and source-specific SQL tools for one chat run.
- `ai/ai_utils/chat_agent.py` constructs the shared LangGraph agent and middleware stack.
- `ai/ai_utils/intent_router.py` classifies each turn without splitting conversation memory.
- `ai/ai_tools.py` exposes catalog lookup and the two-step validation and execution tools to the model.
- `ai/ai_utils/validate_sql.py` performs candidate-based multi-source validation.
- `ai/sql_guard.py` enforces deterministic read-only and allowlist policy.
- `ai/ai_utils/streaming.py` translates graph events into Datastar SSE patches.
- `ai/ai_utils/structured_output_blocks.py` builds backend-owned result blocks.
- `core/db_config/` contains the validated source registry and database connector routing.
- `tests/` mirrors the application modules with focused deterministic coverage.

## Development checks

Most tests mock model and database boundaries, so they do not require an OpenAI API key or a live ClickHouse or SQL
Server instance.

On Windows, use the repository virtual environment when it is available:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe manage.py check
```

On macOS or Linux:

```console
.venv/bin/python -m pytest -q
.venv/bin/python manage.py check
```

Start with the narrowest test module that covers a change, then run the broader suite when the risk warrants it.

## Operate and debug

- [Application logging](app-logging.md) covers log files, retention, identifiers, common events, and debugging.
- [User-facing progress messages](progress-messages.md) explains the status updates emitted during longer agent runs.
- [Changelog](CHANGELOG.md) records visible product and developer-facing changes.

