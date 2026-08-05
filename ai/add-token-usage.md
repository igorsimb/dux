# Token Usage Observability Plan

Goal: add small, unobtrusive token/cost observability to the chat UI without overengineering it.

Scope for this plan:
- Show a compact badge in the top-right area of `ai/templates/ai/ai_main.html`, before the `Подсказки` button.
- On hover, show a breakdown for cumulative session totals: input tokens, output tokens, cost.
- Assume OpenAI-only for now, with `gpt-5.4` as the active model.
- Implement in phases, and stop after each phase until you explicitly approve the next one.

Non-goals for now:
- No multi-provider pricing abstraction.
- No historical analytics, dashboards, or persistence across browser sessions.
- No admin/reporting UI.

## Design Summary

Smallest useful product shape:
- A small badge near the top-right controls, visually quiet but always visible.
- Badge shows cumulative session token total, for example: `1.8k tok`.
- Hover/focus reveals a tooltip/popover with:
  - `In: ...`
  - `Out: ...`
  - `Cost: ...`
- Session means the current page session in the browser, accumulated across chat turns until refresh/reload.

Why this shape:
- It matches your "small, unobtrusive, informative" requirement.
- It avoids cluttering each message bubble.
- It keeps backend scope small: per-request usage is emitted once, frontend accumulates session totals.

## Assumptions

- Token usage should be tracked per user request and accumulated client-side into session totals.
- Use LangChain/LangSmith-compatible `usage_metadata` as the backend contract.
- LangSmith is not part of this implementation plan.
- Cost should be computed locally in the backend from a tiny hard-coded OpenAI pricing table.
- Start with `gpt-5.4` pricing only, and expand only when another model is actually used.
- Keep the app-side implementation thin: emit per-request usage, then accumulate session totals in the frontend.
- If usage metadata is temporarily unavailable for a request, the UI should keep prior totals and simply not increment for that turn.

Pricing source note:
- Pricing values should be synced manually from OpenAI docs when needed.
- Keep a short note near the pricing table with the source links:
  - `https://openai.com/api/pricing/`
  - `https://developers.openai.com/api/docs/pricing`

## Phase 1 - Capture per-request usage on the backend

Goal:
- Reliably capture total input/output tokens for one chat request, including multi-step LangGraph runs.

Implementation:
- Touch `ai/ai_utils/chat_runtime.py`
  - initialize the chat model with streaming usage enabled for OpenAI.
- Touch `ai/ai_utils/streaming.py`
  - add request-scoped usage aggregation for the whole graph run.
  - after the run finishes, emit one structured usage event into the existing queue.
- Touch `ai/views.py`
  - accept the new queue event type and patch Datastar signals with per-request usage payload.

Recommended technical approach:
- Prefer LangChain usage metadata / callback-based aggregation over parsing streamed text.
- Keep the payload shape compatible with LangSmith `usage_metadata`.
- Keep aggregation at request scope, not message-chunk scope.
- Emit one final usage payload shaped roughly like:

```json
{
  "usage_metadata": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801
  },
  "model": "gpt-5.4"
}
```

Verify:
- One normal request produces a usage event.
- A request with tool calls/retries still produces one total covering the full run.
- Existing text streaming behavior stays unchanged.

Testing:
- Add or update focused backend tests for usage aggregation logic.
- If the aggregation is extracted into a helper, test the helper directly.
- If no suitable automated test seam exists yet, create the smallest one rather than relying only on manual verification.

Stop after phase:
- Do not build UI yet.
- Share what payload shape was implemented and confirm it looks correct before proceeding.

## Phase 2 - Add local cost calculation

Goal:
- Compute per-request cost immediately in the backend using the active model and usage metadata.

Implementation:
- Touch `ai/ai_utils/runtime_config.py`
  - reuse `get_model_name()` for model labeling and pricing lookup.
- Touch `ai/ai_utils/streaming.py` or add a small helper such as `ai/ai_utils/token_usage.py`
  - add a tiny pricing table for `gpt-5.4`.
  - compute input, cached input, output, and total cost from `usage_metadata`.
  - include `cost_usd` in the emitted usage event.

Recommended technical approach:
- Use `usage_metadata.input_tokens`, `usage_metadata.output_tokens`, and `usage_metadata.total_tokens` as the base values.
- If `usage_metadata.input_token_details.cache_read` is present, apply cached-input pricing to that subset and normal input pricing to the remainder.
- If detailed cached-input usage is missing, price all input tokens at the standard input rate.
- Keep the pricing table minimal and explicit rather than building a generic pricing system.

Suggested payload shape:

```json
{
  "usage_metadata": {
    "input_tokens": 1234,
    "output_tokens": 567,
    "total_tokens": 1801,
    "input_token_details": {
      "cache_read": 100
    }
  },
  "model": "gpt-5.4",
  "cost_usd": 0.0116
}
```

Initial pricing to encode:
- `gpt-5.4`
  - input: `$2.50 / 1M tokens`
  - cached input: `$0.25 / 1M tokens`
  - output: `$15.00 / 1M tokens`

Verify:
- Cost is available immediately for normal requests.
- Cached-input pricing is used when `cache_read` details are present.
- Missing `cache_read` details fall back cleanly to standard input pricing.

Testing:
- Add unit tests for pricing math.
- Cover normal input/output pricing, cached-input pricing, zero-token cases, and unknown-model fallback.

Stop after phase:
- Do not add the visible badge yet.
- Confirm the computed numbers look correct before wiring UI.

## Phase 3 - Add the compact badge in the top-right header area

Goal:
- Render the small badge before `Подсказки` and keep it visually unobtrusive.

Implementation:
- Touch `ai/templates/ai/ai_main.html`
  - add Datastar signals for session totals.
  - add a compact badge element before the help button.
  - update signals when per-request usage events arrive.

Smallest UI shape:
- Closed state: a subtle badge such as `1.8k tok`.
- Placement: same top-right control row, directly before `Подсказки`.
- Hidden or muted-zero state before first successful usage payload.

Recommended behavior:
- Accumulate session totals in frontend signals:
  - `sessionInputTokens`
  - `sessionOutputTokens`
  - `sessionTotalTokens`
- Keep the badge keyboard-focusable, not hover-only.

Verify:
- Badge appears in the intended position.
- Long chats update cumulative totals correctly.
- Mobile layout remains clean and does not push `Подсказки` awkwardly.

Testing:
- Add the smallest practical template/UI test coverage for signal-driven rendering if the repo already has a pattern for it.
- If there is no existing frontend test setup, verify this phase manually in browser and record the manual checks performed.

Stop after phase:
- Do not add the visible badge yet.
- Confirm the compact badge look before adding hover breakdown polish.

## Phase 4 - Add hover/focus breakdown details

Goal:
- Show the session breakdown on hover/focus: input, output, cost.

Implementation:
- Touch `ai/templates/ai/ai_main.html`
  - add tooltip/popover markup using existing DaisyUI/Tailwind patterns already present in the app.

Tooltip content:
- `In: <formatted input tokens>`
- `Out: <formatted output tokens>`
- `Cost: $<formatted usd>`

Recommended formatting:
- Tokens: compact human-readable display in the badge, fuller numbers in the tooltip.
- Cost: fixed small-dollar precision, e.g. `$0.0042` or `$0.01` depending on magnitude.

Verify:
- Hover works with mouse.
- Focus works with keyboard.
- Tooltip does not obscure the help button or overflow badly on mobile.

Testing:
- Prefer automated coverage for any formatting helper used by the tooltip.
- Manually verify hover and keyboard focus behavior in browser.

Stop after phase:
- This is the first complete user-facing version.

## Phase 5 - Hardening and documentation

Goal:
- Clean up edge cases and document the behavior.

Implementation:
- Touch `README.md`
  - briefly mention token/cost session badge and what it represents.
- Touch tests to close remaining gaps from earlier phases.

Edge cases to handle:
- No usage payload returned.
- Unknown model pricing.
- Multiple requests in one page session.
- Streaming disabled fallback path via `agent.invoke(...)`.

Verify:
- Both streaming and non-streaming paths still work.
- README description matches actual behavior.

Testing:
- Run the relevant test subset added in earlier phases.
- Run the broader affected test suite if present.

## File-by-file implementation map

- `ai/templates/ai/ai_main.html`
  - add session-total signals
  - add compact badge before `Подсказки`
  - add hover/focus breakdown UI
- `ai/views.py`
  - patch new Datastar signals from backend usage events
- `ai/ai_utils/chat_runtime.py`
  - enable usage capture on model init if needed
- `ai/ai_utils/streaming.py`
  - aggregate per-request usage
  - emit usage event into queue
  - compute/request-scoped cost payload
- `ai/ai_utils/runtime_config.py`
  - reuse `get_model_name()` for model labeling in emitted payloads if needed
- `ai/ai_utils/token_usage.py` (optional new helper)
  - tiny pricing table and cost calculation
- `README.md`
  - short note after the feature is complete
