"""CLI demo for capability-based NL->SQL execution with deterministic guardrails.

This demo is intentionally single-source and currently wires only ClickHouse.
Use the web chat runtime for mixed-source sessions across ClickHouse and MSSQL.

Flow:
1) Build SQL toolkit tools and replace `sql_db_query` with `GuardedQuerySQLDatabaseTool`.
2) Register the guarded query tool for `execute_validated_sql` via `set_run_query_tool_for_source`.
3) Expose tools to the model: table discovery/metadata + `validate_sql` + `execute_validated_sql`.
4) Run LangGraph loop: `decide_next_action` -> `execute_tool_calls` -> `decide_next_action`.
   - `decide_next_action` is the LLM node: it reads conversation + tool outputs and decides either
     (a) call the next tool or (b) return a final user answer.
5) `validate_sql` enforces read-only policy, resolves/validates allowed tables,
    rewrites SQL into a normalized, fully-qualified form (db.table), and stores `{validated_id -> sql}` in graph state.
6) `execute_validated_sql` executes only SQL referenced by `validated_id`, marks token as used,
   and returns DB results.

Key invariant: raw model SQL is never executed directly.
Execution is allowed only when `validate_sql` first creates a `validated_id`, and
`execute_validated_sql` then uses that exact `validated_id` to fetch SQL from state.
"""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ai import ai_prompts
from ai.ai_utils.sql_tools import build_guarded_sql_tools, set_run_query_tool_for_source
from ai.ai_tools import (
    execute_validated_sql,
    get_table_descriptions,
    get_table_metadata,
    validate_sql,
)
from core.db_config.source_database_router import get_sql_database_for_source

load_dotenv()

DEMO_SOURCE_ID = "clickhouse_default"


class ValidatedQueryRecord(TypedDict):
    sql: str
    created_at: float
    expires_at: float
    thread_id: str
    status: Literal["validated", "executed"]


class State(MessagesState):
    validated_queries: dict[str, ValidatedQueryRecord]


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.2")


def main() -> None:
    db = get_sql_database_for_source(DEMO_SOURCE_ID)
    model = init_chat_model(get_model_name())

    _sql_tools, list_tables_tool, get_schema_tool, run_query_tool = (
        build_guarded_sql_tools(db, model)
    )
    set_run_query_tool_for_source(DEMO_SOURCE_ID, run_query_tool)

    tools = [
        list_tables_tool,
        get_schema_tool,
        get_table_descriptions,
        get_table_metadata,
        validate_sql,
        execute_validated_sql,
    ]

    def decide_next_action(state: State):
        """LLM decision node: consume current messages and either call tools or finish with final answer."""
        system_message = {
            "role": "system",
            "content": ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC,
        }
        llm_with_tools = model.bind_tools(tools)
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("decide_next_action", decide_next_action)
    # docs: https://docs.langchain.com/oss/python/langchain/tools#toolnode
    builder.add_node("execute_tool_calls", ToolNode(tools))

    builder.add_edge(START, "decide_next_action")
    builder.add_conditional_edges(
        "decide_next_action",
        tools_condition,
        {"tools": "execute_tool_calls", "__end__": END},
    )
    builder.add_edge("execute_tool_calls", "decide_next_action")

    agent = builder.compile(checkpointer=InMemorySaver())

    config: RunnableConfig = {
        "configurable": {"thread_id": "demo-thread"},
        "recursion_limit": 30,
    }
    question = "Покажи десять клиентов с наибольшей суммой заказов за последние 30 дней."

    for step in agent.stream(
        {"messages": [HumanMessage(content=question)], "validated_queries": {}},
        config,
        stream_mode="values",
    ):
        if "messages" in step and step["messages"]:
            # Note: if model makes multiple tool calls, we only print the last one
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
