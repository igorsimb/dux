# Intent routing for one shared LangChain agent

This app has one chat surface, but users do not always ask for the same kind of help.

Some turns are real data questions:

- `show the top 10 customers by revenue`
- `compare sales across product categories`

Some are much lighter:

- `what can you do?`
- `hi`
- `switch the theme to nord`

If every message went through the full SQL-oriented prompt and the full tool list, the app would spend extra tokens,
carry unnecessary tool context, and make simple requests harder than they need to be. At the same time, this project
still needs one continuous conversation, one history, and one place to observe what happened.

That is why the app uses a single LangChain `create_agent(...)` agent with deterministic per-turn routing.

## Why this matters in this app

This assistant is primarily a guarded SQL assistant. The SQL path is the important path: it has the business-focused
prompt, the tool set, and the safety constraints needed for data questions.

But the same UI also supports lightweight turns around that core experience:

- a quick meta question like `what can you do?`
- casual smalltalk like `hi`
- UI actions like `switch the theme to nord`

The routing layer keeps those turns simple without breaking the main SQL workflow.

## How it works

The high-level flow is straightforward:

1. The app builds one shared agent in `ai/ai_utils/chat_agent.py` via LangChain `create_agent(...)`.
2. Before each model call, `ai/ai_utils/intent_middleware.py` inspects the latest user message.
3. The message is classified by deterministic code in `ai/ai_utils/intent_router.py`.
4. For that turn only, middleware overrides the system prompt and available tools.
5. The same agent, state schema, checkpointer, and conversation history continue as usual.

So the routing decision changes the current turn's behavior, not the overall agent architecture.

## Routing modes

The router currently selects one of three modes:

- `smalltalk_meta`
- `theme_change`
- `sql_agent`

The classifier is intentionally conservative. If a message is unclear, mixed, or looks even partly like a business or
data request, it goes to `sql_agent`.

In practice, the precedence is:

1. business/data indicators -> `sql_agent`
2. theme change -> `theme_change`
3. smalltalk or meta -> `smalltalk_meta`
4. everything else -> `sql_agent`

That means a mixed turn like `hi, show the top 10 customers by revenue` stays on the SQL path.

## What each mode does

### `smalltalk_meta`

Use this for greetings and lightweight questions about the assistant itself.

Examples:

- `hi`
- `hello there`
- `what can you do?`

For this mode, middleware swaps in `SYSTEM_PROMPT_SMALLTALK_META` and exposes no tools.

That keeps the reply short and human, instead of dragging the model through the full SQL tool context for a message
that does not need it.

### `theme_change`

Use this for UI theme requests.

Examples:

- `switch the theme to nord`
- `change the theme`
- `make the theme lighter`

For this mode, middleware swaps in `SYSTEM_PROMPT_THEME_CHANGE` and exposes only one tool:

- `switch_color_theme`

This matters because the model does not need access to the SQL tool set just to change the interface theme.

When a theme change succeeds, the user sees two things:

- a short acknowledgement message; this doc uses an English example like `Done — switched the theme to nord.`,
  but the actual app still requires user-facing replies to be in Russian
- a UI theme signal that the frontend can apply immediately

### `sql_agent`

This is the default and the safe fallback.

Examples:

- `show the top 10 customers by revenue`
- `compare sales across product categories`
- `stock levels for product 12345`
- `hi, show the largest orders this month`

For this mode, middleware keeps the guarded SQL prompt and exposes only SQL workflow tools. The theme tool remains
registered with the shared agent but is hidden from SQL model calls.

This is also where unclear requests land. That bias is deliberate: if there is any chance the user is asking for real
business data, the app prefers the SQL-capable path over a lightweight conversational one.

## Why not two agents

An earlier idea was to put a lightweight router agent in front of the SQL agent.

That approach was rejected for practical reasons:

- it added another model hop for a problem handled well by deterministic code
- it made the architecture harder to debug and reason about
- it risked split-memory behavior between lightweight turns and SQL turns

The current design keeps one agent and changes only the prompt and tool exposure for the current turn.

## Memory, history, and conversation continuity

Routing does not create separate conversations.

All modes share the same:

- LangChain agent instance
- `ChatAgentState`
- checkpointer-backed thread state
- conversation history

History trimming still runs before model calls, and the same checkpointer is reused across turns. In practice, a user
can ask `what can you do?`, then `switch the theme to nord`, then `show the top 10 customers by revenue` in one
continuous chat without hopping between different agent memories.

## Logging, observability, and usage accounting

Each routed turn is logged with:

- `chat.intent selected mode=...`

That gives a compact signal in logs about how the turn was handled, without logging raw user text in the routing log
line.

Usage accounting still works for lightweight turns too:

- token and cost badges continue to update for `smalltalk_meta`
- token and cost badges continue to update for `theme_change`

So even when no SQL tools run, the session usage indicators stay accurate.

## Where to look in the code

If you want to trace the implementation, these are the main files:

- `ai/ai_utils/chat_agent.py` - builds the single `create_agent(...)` agent and wires in middleware
- `ai/ai_utils/intent_router.py` - deterministic intent classification and precedence
- `ai/ai_utils/intent_middleware.py` - per-turn prompt/tool overrides before model calls
- `ai/views.py` - chat streaming path, including theme signals and frontend-visible updates
- `tests/ai/test_chat_agent.py` - middleware behavior, prompt/tool overrides, and routing logs
- `tests/ai/test_views_streaming_flow.py` - streamed theme acknowledgement and theme signal behavior
- `tests/ai/test_views_usage_signals.py` - usage badge updates for theme turns

## Quick summary

The intent router is intentionally simple:

- one LangChain agent
- deterministic routing code
- per-turn middleware overrides
- conservative fallback to `sql_agent`
- shared memory and history across everything

That keeps the SQL assistant safe and capable, while making smalltalk and UI actions feel lighter and more natural.
