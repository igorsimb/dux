"""Custom chat agent state for validated SQL tokens and SQL render state."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain.agents import AgentState
from typing_extensions import NotRequired


class ValidatedQueryRecord(TypedDict):
    sql: str
    source_id: str
    dialect: str
    tables: list[str]
    created_at: float
    expires_at: float
    thread_id: str
    status: Literal["validated", "executed"]


class ChatAgentState(AgentState):
    validated_queries: NotRequired[dict[str, ValidatedQueryRecord]]
    model_response_layout: NotRequired[dict[str, Any] | None]
    sql_result_table_blocks: NotRequired[list[dict[str, Any]]]
