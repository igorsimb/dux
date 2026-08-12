# Changelog

## 2026-07-06 (`v0.9.35`)

- SQL result tables now include a permission-gated `Details` panel with source, table, row-count, model-note, and
  raw-SQL context.
- Authorized users can expand and copy formatted raw SQL from `Show SQL`; unauthorized users do not receive restricted
  details in the page HTML or saved block JSON.
- Same-tab restore now rebuilds permitted table details from the saved structured block data.

## 2026-06-15 (`v0.9.3`)

- SQL result tables now show large numeric values with readable spacing and decimal commas, making quantities and sales
  amounts easier to scan.
- SQL generation now prefers human-readable English aliases for final result columns, so copied tables have clearer
  headers for metrics like moving averages and daily percentage changes.

## 2026-06-13

- SQL result answers now render as structured `data_table` blocks instead of model-generated markdown tables.
- Table rows are built from backend SQL execution results, while the model only provides commentary and table placement.
- Live chat renders structured blocks through Django partials, and same-tab restore rebuilds them from saved raw block
  JSON.
- Empty SQL results render as empty tables; malformed or unmatched table blocks are skipped with warnings instead of
  breaking the response.
- Added documentation for the structured-output block flow and verified the full suite with `320 passed`.

## 2026-06-08 (`v0.9.25`)

- Assistant replies now render GitHub Flavored Markdown in the chat UI, so lists, code blocks, and SQL-result tables
  are easier to read.
- Markdown output is sanitized before rendering, and the chat falls back to safe plain text if CDN libraries are
  unavailable.
- Refreshed same-tab conversations preserve the raw markdown source in the existing session transcript.

## 2026-06-03 (`v0.9.2`)

- Ambiguous SQL requests now pause for clarification instead of guessing missing details like date ranges or ranking criteria.
- The chat resumes the saved agent run after the user answers the clarification question, keeping the flow in the same conversation.
- Added documentation for the HITL clarification flow and upgraded the LangChain stack to use built-in `respond` decisions.

## 2026-03-30 (`v0.9.12`)

- Added persistent application logs under `logs/` so server-side debugging no longer depends on entering the container.
- Split operational app logs from per-chat transcript files, making it easier to inspect request flow and user-visible chat history separately.
- Error and timeout messages now include a short conversation code that support can use to locate the matching transcript faster and narrow the related backend log window.
- Chat transcript files now keep their existing filenames but include a one-time conversation code header inside the file.

## 2026-03-27 (`v0.9.1`)

- Greetings and simple meta questions now return faster without going through the full SQL flow.
- Theme-change requests are handled more directly and now switch the UI theme with a short confirmation reply.
- Chat memory stays consistent when switching between casual messages, theme changes, and SQL questions.
- Session token and cost badges continue updating correctly on these lighter chat turns.

## 2026-03-24

- Added plain-text per-chat transcript logging under `chatlogs/<username>/...` for completed visible turns.
- New chat sessions now write to separate transcript files derived from the backend chat thread identity.

## 2026-03-19 (`v0.9.0`)

- Chat now restores the visible transcript and cumulative usage badge after refresh in the same browser tab.
- Added a `New chat` button that rotates the tab session key and resets the visible chat and usage counters.

## 2026-03-14 (`v0.8.8`)

- Chat now remembers the current conversation across messages and page refreshes in the same browser tab.
- Starting a new browser tab still starts a separate chat session.
- Short-term memory is kept more stable on the backend, while older context is trimmed before model calls to stay within budget.
- Session token totals in the UI remain informational and are shown independently from backend memory.

## 2026-03-14 (`v0.8.7`)

- Chat replies are more reliable, especially when a response arrives late or streaming is interrupted.
- Waiting and timeout behavior is clearer, so the chat is less likely to feel stuck or show confusing updates.
- Message delivery is smoother and more consistent, with cleaner final responses in edge cases.

## 2026-03-11

- Added session token and cost observability to the chat UI.
- Added a compact usage badge with a hover/focus breakdown for session totals.
