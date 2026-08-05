# HITL clarification flow for ambiguous SQL requests

Users often ask useful but incomplete data questions.

For example:

- `show me recent transactions`
- `покажи последние продажи`
- `top customers by revenue`

Those requests are risky for a SQL assistant because words like `recent`, `последние`, and `top` hide parameters that
change the query result. Guessing a period, ranking metric, or required date filter can produce a confident answer that
looks precise but is based on an assumption the user never approved.

This app handles that case with a clarification-first human-in-the-loop flow.

## Why this matters in this app

The assistant can execute read-only SQL through the guarded validation flow, so ambiguity needs to be handled before SQL
is drafted and validated. The existing `validate_sql` and `execute_validated_sql` tools protect the database execution
path, but they cannot know whether the user's business intent was complete.

The clarification flow fills that gap. When the model needs a missing parameter, it calls `ask_user` instead of
guessing. LangChain's `HumanInTheLoopMiddleware` pauses the graph, the app shows the question to the user, and the next
message in the same chat resumes the saved graph state.

## How it works

The flow is intentionally small:

1. The SQL prompt in `ai/ai_prompts.py` tells the model to call `ask_user(question)` for ambiguous SQL inputs.
2. The `ask_user` tool is defined in `ai/ai_tools.py` and included in the SQL tool list from
`ai/ai_utils/chat_runtime.py`.
3. `ai/ai_utils/chat_agent.py` wires `HumanInTheLoopMiddleware` to interrupt only on `ask_user`.
4. `ai/ai_utils/streaming.py` detects a pending `ask_user` interrupt and shows the tool's `question` as assistant text.
5. The next user message in the same chat thread is sent as a `respond` decision, so the graph continues from the saved
checkpoint.

The user does not see a separate approval UI. From the chat surface, it looks like a normal clarification question.

Example:

```text
User: покажи последние продажи
Assistant: За какой период показать продажи?
User: за последние 30 дней
Assistant: ...runs the validated SQL flow and returns the answer...
```

## What counts as ambiguous

The current implementation relies on the SQL prompt and the configured GPT-5.x model behavior rather than a deterministic
phrase detector. That keeps the backend from blocking valid queries with brittle text rules.

The prompt specifically tells the model to ask before querying when the user omits:

- a concrete period for ambiguous timeframe words like `recent`, `latest`, `fresh`, `актуальные`, or `последние`
- a ranking count or metric for ambiguous ranking words like `top`, `best`, `лучшие`, or `топ`
- a concrete date range for tables whose metadata has `requires_date_filter=true`

The existing default result limit still applies to ordinary non-ranking requests. If the only missing detail is output
row count, the assistant may use the default limit rather than interrupting.

## Resume behavior

The app always treats the next message in the same chat session as the answer to a pending `ask_user` interruption.

In code, `ai/ai_utils/streaming.py` checks the checkpointed thread state before starting a run. If the thread is paused
on `ask_user`, it sends:

```python
Command(
    resume={
        "decisions": [
            {"type": "respond", "message": user_text},
        ]
    }
)
```

The `message` field is important: LangChain's Python HITL `respond` decision returns that value as the synthetic tool
result. The app does not use the older issue-snippet key `content`.

If the model ever emits multiple `ask_user` calls in one interrupt, the app joins the questions into one assistant
message and uses the next user input as the `respond` message for each pending action. The configured model is prompted
to ask one concise Russian question, so multiple simultaneous questions should be unusual.

## Persistence limits

The app keeps using the shared `InMemorySaver` from `ai/ai_utils/checkpointer.py`.

That is enough for the current request flow because separate HTTP requests in the same Django process can reuse the same
`thread_id` and resume a paused graph. It has two important limits:

- interrupted conversations do not survive process restart
- interrupted conversations are not shared across multiple Django worker processes

A production deployment that needs durable pauses should replace this with a persistent LangGraph checkpointer, such as
`AsyncPostgresSaver`, and handle the database connection lifecycle explicitly.

## Where to look in the code

The main files are:

- `ai/ai_tools.py` - defines `ask_user`, `validate_sql`, and `execute_validated_sql`
- `ai/ai_prompts.py` - describes when the model must ask instead of guessing
- `ai/ai_utils/chat_runtime.py` - exposes `ask_user` in the SQL tool list
- `ai/ai_utils/chat_agent.py` - configures `HumanInTheLoopMiddleware` and the shared checkpointer
- `ai/ai_utils/streaming.py` - extracts interrupts, displays clarification questions, and builds resume commands
- `ai/views.py` - keeps the existing chat UI flow; no separate HITL UI is required

Relevant tests:

- `tests/ai/test_chat_agent.py`
- `tests/ai/test_chat_runtime_build_model_and_tools.py`
- `tests/ai/test_ai_prompts.py`
- `tests/ai/test_streaming_usage.py`

## Why this approach

This change keeps SQL execution automatic after validation and uses HITL only for missing user intent. That matches the
actual problem: the unsafe behavior was not SQL execution without validation, but the model inventing missing parameters
before it reached validation.

The result is a small, stateful clarification loop that preserves the existing chat experience while avoiding the most
expensive kind of assumption: a precise-looking answer to an under-specified business question.
