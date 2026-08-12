# User-facing progress messages

Long SQL turns should not feel stuck.

This app often has to inspect table descriptions, read table metadata, validate generated SQL, wait for a database, and
then let the model explain the result. Even when everything is working, that can take several seconds. The progress
message system gives the user a small, changing hint about what Dux is doing while the final answer is still being
prepared.

These messages are user-facing UI hints, not backend diagnostics. They make the waiting state easier to understand
without exposing raw SQL, tool payloads, or operational details.

## How it works

The path from a tool stage to the waiting alert is short:

```text
SQL/tool stage
-> ai/ai_utils/progress_messages.py picks an English phrase
-> show_progress_message(...) writes it with LangGraph get_stream_writer()
-> LangGraph emits it on the custom stream
-> ai/ai_utils/streaming.py turns custom text into {kind: "progress", text: ...}
-> ai/views.py patches the Datastar UserFacingProgressMessage signal
-> ai/templates/ai/ai_main.html renders the waiting alert
```

The visible alert is the row under the chat transcript while `isWaitingResponse` is true:

```text
Dux <UserFacingProgressMessage> <thinkingSeconds> sec.
```

The default signal value is `is thinking...`. A real SQL turn may replace it with phrases like
`is checking the SQL query`, `is waiting for the database response`, or `is preparing the final answer`, depending on
which tool stage emitted the latest custom event.

## What stages exist

The implementation groups messages around the practical steps a user is waiting on:

- table catalog lookup, when the agent checks which allowlisted tables are available
- table metadata lookup, including found and not-found outcomes
- SQL validation, including retry hints when a query is rejected and rewritten
- validated SQL execution, including token problems, DB connection/configuration issues, waiting for the database,
  database errors, and final result analysis

Each stage has a set of short English phrases. The helper chooses one at random at runtime, so repeated turns do not
show the exact same wording every time.

## Why the phrases are randomized

The phrases are written as present-continuous fragments that fit after `Dux` in the waiting alert. This keeps every
stage grammatically consistent, such as `Dux is checking available tables`.

Randomization is deliberately lightweight. It does not change control flow or stage semantics; it only avoids making
long-running turns feel mechanical when the same stage happens often, such as SQL validation retries or DB waiting.

## Limits of this system

Progress messages are best-effort hints about the current chat flow. They are not durable records and they are not meant
to explain every backend decision.

Use them for user experience, not for incident analysis:

- they are streamed UI status text, not an audit log
- they can be superseded by later progress messages in the same turn
- they do not replace `logs/debug.log`, `logs/error.log`, or per-chat transcripts
- they do not provide final answer provenance; the final response and structured SQL result blocks remain the source of
  what the user actually received

For operational debugging, use the logging flow in `docs/app-logging.md`.

## Where to look in the code

The main implementation files are:

- `ai/ai_utils/progress_messages.py` - stage enums, randomized phrase sets, fallback text, and
  `show_progress_message(...)`
- `ai/ai_tools.py` - tool calls that emit progress around table lookup, metadata lookup, SQL validation, SQL execution,
  DB waiting/error handling, and final analysis
- `ai/ai_utils/streaming.py` - LangGraph `custom` stream handling and progress queue events
- `ai/views.py` - queue handling for `kind == "progress"` and Datastar signal patches
- `ai/templates/ai/ai_main.html` - `UserFacingProgressMessage` signal defaults and the waiting alert markup

Related behavior is documented in `docs/structured-output-blocks.md` for final table rendering and
`docs/app-logging.md` for operational logs.
