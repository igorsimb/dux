# Application Logging

The app writes two kinds of operational text on purpose.

`logs/` is for backend behavior: request flow, agent execution, SQL validation, SQL execution, usage, and failures.
`chatlogs/` is for readable conversation transcripts: what the user and assistant actually saw.

Keeping these separate matters because support work usually starts with a human scanning live backend logs, then pivots
to a transcript only when the user-visible conversation matters.

## What gets written where

Application logs are configured in `ai/ai_utils/logging_config.py`.

They are written to:

- `/app/logs/debug.log`
- `/app/logs/error.log`

In Docker Compose, that directory is exposed to the host through `docker-compose.yml`:

```yaml
volumes:
  - ./logs:/app/logs
```

So on the server, you can inspect logs directly in `./logs/` without entering the container.

Chat transcripts are written under `chatlogs/<username>/...` from `ai/ai_utils/chat_logging.py`.

## How logging is initialized

The app uses Loguru directly, so logging is initialized explicitly in the active startup paths instead of routing
everything through Django's standard `LOGGING` setting first.

Current startup points:

- `manage.py`
- `config/asgi.py`

This keeps the implementation small and predictable.

## Log files and retention

The current config lives at the top of `ai/ai_utils/logging_config.py`:

```python
LOG_DIR = "logs"
LOG_LEVEL = "DEBUG"
LOG_ROTATION = "100 MB"
LOG_RETENTION = "14 days"
LOG_JSON = False
```

The app keeps stable active filenames for day-to-day use:

- `debug.log` for `DEBUG` and above
- `error.log` for `ERROR` and above

When Loguru rotates files, archived filenames include timestamps automatically. That gives you a stable current file
plus dated history without manual cleanup.

## Live log format

Backend logs use one human-readable structured text format:

```text
11:10:56.065 DEBUG chat.request     ready              thread=5f822109 user_len=11
```

Read each line as:

```text
TIME         LEVEL AREA             EVENT              DETAILS
```

The important behavior appears before the details. Details are stable `key=value` pairs, so a human can scan the stream
and an LLM can still reconstruct what happened when the logs are pasted into a chat.

Long secondary details use indented lines:

```text
11:10:58.062 DEBUG chat.message     ai_requested_tools index=7 count=2
           1. get_table_metadata args=table call=call_0m0Wx3n
           2. get_table_metadata args=table call=call_vyFFKGk
```

SQL remains multiline and copy/paste-friendly:

```text
11:21:20.484 DEBUG tool.sql         query_ready        thread=0bb312ff validated_id=d809f071
SELECT
  COUNT(*) AS count
FROM analytics.customers
-- end sql
```

## Conversation identifiers

The canonical backend identifier is `thread_id`.

It is derived on the server from:

- authenticated user
- Django session
- normalized browser `chatSessionKey`

Full thread IDs are long, so most live log lines show a short stable display id:

```text
thread=5f822109
```

That value comes from the digest portion of the canonical thread id. It is meant for live scanning and correlation
across nearby log lines.

The app also derives a short public conversation code from `thread_id` in `ai/ai_utils/logging_config.py`. That code is
used for support workflows:

- it is shown to the user only on error and timeout paths
- it is added to transcript content as a one-time header

This split keeps backend logs compact while still giving users a short code they can report.

## Area names

The `AREA` column names the subsystem that emitted the event.

- `chat.input`: user input received by the backend.
- `chat.request`: one SSE request lifecycle, including cancellation, cleanup, timeout, and HITL waiting.
- `chat.agent`: agent setup, start, completion, and runtime failures.
- `chat.graph`: checkpoint inspection, graph input preparation, state snapshots, and interrupts.
- `chat.resume`: human-in-the-loop resume command construction and confirmation.
- `chat.message`: new AI/tool messages observed in graph state.
- `chat.memory`: message trimming decisions before model calls.
- `chat.intent`: intent routing decisions.
- `chat.usage`: token usage and cost.
- `tool.metadata`: table-description and table-metadata tools.
- `tool.sql`: SQL validation, candidate routing, SQL execution, and SQL rejections.
- `tool.ui`: theme switching and structured response layout tools.
- `chat.transcript`: transcript file write failures.
- `db.clickhouse`: ClickHouse connection setup diagnostics.

## Common events

Request and agent flow usually starts like this:

```text
11:10:56.063 DEBUG chat.input       received           len=42 preview="..."
11:10:56.065 DEBUG chat.request     ready              thread=5f822109 user_len=42
11:10:56.097 DEBUG chat.agent       tools_ready        count=7
11:10:56.118 DEBUG chat.agent       started            thread=5f822109 streaming=true
```

Graph and message events show how the agent run progresses:

```text
11:10:56.116 DEBUG chat.graph       input_ready        mode=fresh_user_message
11:10:56.117 DEBUG chat.graph       baseline_ready     messages=6 layout=false sql_blocks=0
11:10:58.062 DEBUG chat.graph       state_received     messages=8 layout=false sql_blocks=0 interrupts=0
11:10:58.062 DEBUG chat.message     ai_requested_tools index=7 count=1
           1. validate_sql args=query call=call_0m0Wx3n
```

SQL events identify validation and execution:

```text
11:21:20.100 DEBUG tool.sql         validation_started thread=0bb312ff query_len=742
11:21:20.210 DEBUG tool.sql         validation_passed  thread=0bb312ff validated_id=d809f071 tables=analytics.customers
11:21:20.484 DEBUG tool.sql         query_ready        thread=0bb312ff validated_id=d809f071
SELECT
  COUNT(*) AS count
FROM analytics.customers
-- end sql
11:21:20.600 DEBUG tool.sql         executed           thread=0bb312ff validated_id=d809f071 result_type=list
           result_len=183
```

Usage events identify the model accounting for a completed run:

```text
11:11:00.407 DEBUG chat.usage       raw_received       model=gpt-5.4 in=7487 out=128 total=7615 cache=6400
11:11:00.408 INFO  chat.usage       recorded           model=gpt-5.4 in=7487 out=128 total=7615 cache=6400 cost=$0.0062
```

HITL clarification shows up as graph interruption and request waiting:

```text
11:11:00.407 DEBUG chat.graph       interrupted        thread=5f822109
11:11:00.408 DEBUG chat.request     hitl_waiting       thread=5f822109
```

## What a transcript looks like

Transcript filenames stay unchanged.

Example path:

```text
chatlogs/i.dolgikh/30-03-2026-e0ddb.txt
```

The file content starts with a stable conversation header:

```text
Conversation: 5f2ab1c0f4d2

[2026-03-30 15:12:41]
User: Привет
AI: Здравствуйте
```

The header is written once when the file is created. Later turns append normally.

## Why the UI error code is separate

User-visible timeout and error messages can include a shorter support code such as:

```text
(Ошибка: не удалось получить ответ. Попробуйте запрос еще раз.) Код разговора: 5f2ab1c0f4d2.
```

That text is built separately from the base persisted transcript/log text.

This is deliberate:

- the UI needs a short code the user can report
- backend logs need compact request correlation
- transcript files should stay readable and should not depend on UI formatting

The relevant code paths are:

- `ai/ai_utils/chat_errors.py`
- `ai/views.py`
- `ai/ai_utils/chat_runtime.py`

## Where to look in code

If you need to change this system later, start here:

- `ai/ai_utils/logging_config.py`: Loguru sinks, file names, rotation, retention, format helpers, and id shortening.
- `docker-compose.yml`: host volume for `/app/logs`.
- `ai/views.py`: request lifecycle logging and user-visible timeout/error rendering.
- `ai/ai_utils/chat_runtime.py`: agent setup, runtime exception handling, and reset retry logs.
- `ai/ai_utils/streaming.py`: graph state, HITL, message, layout, and usage logs.
- `ai/ai_utils/intent_middleware.py`: intent routing logs.
- `ai/ai_utils/memory_middleware.py`: memory trimming logs.
- `ai/ai_tools.py`: metadata tool, UI tool, SQL validation wrapper, and SQL execution logs.
- `ai/ai_utils/validate_sql.py`: candidate-level SQL validation diagnostics.
- `ai/ai_utils/chat_logging.py`: transcript file naming, transcript content format, and transcript write warnings.

## Practical debugging workflow

If a user reports a problem:

1. Ask for the short conversation code shown in the error message.
2. Open the transcript file under `chatlogs/` and look for `Conversation: <code>`.
3. Use that transcript to review the user-visible conversation and narrow the time window.
4. Inspect `logs/debug.log`, `logs/error.log`, or a live `docker logs -f <container>` stream around the same time.
5. Follow the short `thread=<id>` across `chat.request`, `chat.agent`, `chat.graph`, `tool.sql`, and `chat.usage`.
6. If more analysis is needed, paste the relevant contiguous log section into an LLM/agent.

When pasting logs into an LLM, include the first `chat.input received` / `chat.request ready` line, the agent and graph
state lines, any `chat.message ai_requested_tools` detail lines, the SQL block, and the final `chat.usage recorded` or
failure line. That gives enough locality for another agent to reconstruct the turn without needing a separate JSON file.
