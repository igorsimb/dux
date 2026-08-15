# Project Instructions

This file adds project-specific guidance on top of the global agent instructions.

## Project Overview

This project's core is a guarded natural-language-to-SQL system for querying business data. It uses LangChain/LangGraph
and OpenAI models to route each turn, inspect an allowlisted data catalog, generate and validate SQL, execute it against
ClickHouse or Microsoft SQL Server, and return backend-owned structured results. A Python 3.13/Django 6 application
with an authenticated English-first Datastar/SSE chat UI provides the delivery layer around that NL-to-SQL flow.

- Treat `README.md` as the current behavior overview, but verify implementation details in code when they disagree.
- The supported web entry point is `config.asgi:application`, normally run with Daphne. `manage.py` is the Django
  administrative entry point and configures application logging before Django starts.
- Application-owned UI and assistant-facing copy are English. Preserve the focused English/Russian input-routing
  literals, tests, and `docs/intent-router.md` examples that verify deterministic bilingual intent recognition.

## Architecture Map

- `config/settings.py`: Django settings, SQLite application database, custom `core.User`, allauth, and environment
  loading.
- `config/urls.py`, `ai/urls.py`: authentication routes and the login-protected chat page/SSE endpoint.
- `ai/views.py`: async HTTP/SSE lifecycle, request timeouts, permission projection, and UI patch queue.
- `ai/ai_utils/chat_runtime.py`: single async runtime entry point; builds the model and source-specific SQL tools.
- `ai/ai_utils/chat_agent.py`: constructs the one shared LangGraph agent and middleware stack.
- `ai/ai_utils/intent_router.py`, `intent_middleware.py`: deterministic per-turn routing between `smalltalk_meta`,
  `theme_change`, and conservative fallback `sql_agent` without splitting conversation memory.
- `ai/ai_tools.py`: model-visible catalog, validation, and capability-based SQL execution tools.
- `ai/ai_utils/validate_sql.py`, `ai/sql_guard.py`: deterministic read-only, parseability, source, dialect, and table
  allowlist enforcement. Keep policy decisions here rather than in prompts.
- `ai/ai_utils/streaming.py`, `structured_output_blocks.py`, `ui.py`: translate graph events and backend-owned result
  data into Datastar SSE patches. Do not make the model the source of truth for table rows or sensitive details.
- `ai/templates/ai/ai_main.html`: the Datastar client, sessionStorage-backed per-tab transcript, usage display, and
  theme UI.
- `core/db_config/`: validated source registry plus lazy ClickHouse/MSSQL connector routing.
- `tests/ai/`, `tests/core/`, `tests/config/`: focused pytest suites mirroring the modules above.

For deeper context, start with the focused document matching the change: `docs/intent-router.md`,
`docs/sql_candidate_validation_flow.md`, `docs/structured-output-blocks.md`, `docs/progress-messages.md`, or
`docs/app-logging.md`.

## Behavioral Invariants

- Preserve the two-step SQL capability boundary: the model calls `validate_sql(query)`, receives a short-lived,
  thread-bound, one-time `validated_id`, and can execute only through `execute_validated_sql(validated_id)`. Never add a
  model-facing raw-query execution path or trust model-supplied source/dialect/table metadata at execution time.
- SQL validation is multi-source and candidate-based. A query must resolve successfully to exactly one configured
  `(source_id, sql_dialect)` pair. Keep safety, allowlist, and routing checks deterministic and unit-testable without
  an LLM.
- One agent and one conversation state serve every intent. Intent middleware changes prompt/tool exposure per turn;
  `smalltalk_meta` is tool-free and `theme_change` exposes only `switch_color_theme`.
- The normal web path is async end-to-end. Avoid sync wrappers, nested event loops, or blocking database/model work in
  the request lifecycle unless the existing boundary explicitly handles it.
- SQL result rows and answer-detail facts are application-owned structured blocks. The final model message references
  them; it must not reconstruct or authorize their content.
- Raw SQL and model notes are permission-gated through `core.view_raw_sql` and `core.view_answer_notes`. Apply filtering
  before rendering/SSE delivery, not only by hiding frontend elements.
- Chat continuity is scoped by the frontend `chatSessionKey` and the in-process checkpointer. It is short-term memory,
  not durable cross-process storage; visible transcript persistence in sessionStorage is a separate concern.

## Data Source and Catalog Changes

The data catalog is configuration, but it is also part of the SQL security boundary. Keep these files consistent:

- `core/db_config/sql_sources.json`: source id, driver, dialect, default database, and environment-variable mapping.
- `ai/table_descriptions.json`: model-visible allowlisted table rows; each row declares matching `source` and
  `sql_dialect` values.
- `ai/table_metadata.json`: detailed model-facing table/column metadata keyed by table name.

Use `docs/how_to_add_table.md` for the complete workflow and `docs/table_metadata_reference.md` for field semantics.
Configuration loaders fail fast on missing sources, mismatched dialects, and malformed JSON. Preserve that behavior and
add focused loader/routing tests for catalog changes. Do not connect to production databases merely to validate static
configuration when deterministic tests can cover the risk.

## Development and Verification

- Dependencies are declared in `pyproject.toml`; `requirements.txt` and `uv.lock` are generated lock artifacts. Do not
  hand-edit generated dependency files independently of the source dependency change and its lock/compile workflow.
- Prefer the existing Windows virtual environment when present: `.venv/Scripts/python.exe -m pytest <target>`.
  Otherwise use `python -m pytest <target>` with Python 3.13.
- Run the narrowest test module that mirrors the touched code first. Most tests mock model and database boundaries and
  should not require `OPENAI_API_KEY`, ClickHouse, MSSQL, or network access.
- For Django model, permission, authentication, or settings changes, include the relevant `pytest-django` test and run
  `python manage.py check`; create migrations only when the schema or model metadata actually changes.
- For SSE/runtime changes, start with `tests/ai/test_views_streaming_flow.py`, `test_streaming_usage.py`, and the
  focused runtime test matching the changed module. For SQL changes, start with the corresponding `test_validate_sql*`,
  `test_sql_guard.py`, `test_sql_tools_config.py`, or `test_ai_tools_execute_validated_sql.py` coverage.
- Local web smoke testing can use Daphne's ASGI-backed `python manage.py runserver`. The Quickstart setup command
  migrates the SQLite application database before Docker Compose startup. Compose binds host `chatlogs/` and `logs/`;
  do not commit their runtime contents.

## Configuration Boundaries

- `.env` is loaded locally, but secrets and deployment-specific connection values must remain in environment variables.
- `OPENAI_MODEL` selects the model. Trust `ai/ai_utils/runtime_config.py` for the current default rather than
  duplicating a model version in new docs or code.
- Chat execution always uses agent event streaming. Preserve the `astream` event flow and final-state reconciliation
  when changing response handling; do not add a non-streaming execution mode.
- Source JSON maps logical connection fields to environment-variable names. Connector code resolves those names and
  should report missing required configuration precisely without logging secrets.
- Application data uses SQLite; queried business data stays in external ClickHouse/MSSQL sources. Do not confuse Django
  migrations with external warehouse schema management.

## Documentation Style

Write documentation for humans first.

- Keep all project documentation in English, including examples, unless a file is explicitly user-facing product copy.
- Aim for the same style and structure as `docs/intent-router.md`.
- Write in an approachable, technical tone similar to strong LangChain docs: clear, practical, and easy to scan.
- Start with the problem in plain language before describing the implementation.
- Explain why the behavior matters in this app, not just what the code does.
- Prefer short sections with explanatory prose and small lists over dense spec-style bullet dumps.
- Add concrete examples when they help a developer understand routing, behavior, or usage.
- Keep examples realistic and concise.
- Document the actual implementation, not an aspirational design.
- Include exact file paths when pointing readers to the code.
- When relevant, cover four things explicitly: the problem, how it works, why this approach was chosen, and where to look in the code.
- Avoid writing for LLM consumption. Do not use checklist-heavy or machine-oriented phrasing unless the document is specifically a plan.

## Changelog Style

- Keep changelog entries simple, feature-focused, and easy to read.
- Describe visible product behavior and developer-relevant outcomes.
- Avoid architecture-history narration unless it directly matters to users or maintainers.

## Documentation Scope

- `README.md` should stay concise and orienting.
- Architecture docs should feel like implementation guides, not design notes.
- Plan docs may stay more procedural, but reference the implemented docs once work is complete.
