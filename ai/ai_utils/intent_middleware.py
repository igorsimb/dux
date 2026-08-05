from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from loguru import logger

from ai import ai_prompts

from .agent_state import ChatAgentState
from .intent_router import (
    INTENT_SMALLTALK_META,
    INTENT_SQL_AGENT,
    INTENT_THEME_CHANGE,
    classify_intent,
)
from .logging_config import format_log_event


@dataclass(frozen=True)
class IntentRoute:
    mode: str
    prompt: str
    tools: list[Any]


def build_intent_routing_middleware(sql_tools: list[Any], sql_prompt: str):
    return IntentRoutingMiddleware(sql_tools, sql_prompt)


def select_intent_route(
    latest_user_message: str,
    sql_tools: list[Any],
    theme_tools: list[Any],
    sql_prompt: str,
) -> IntentRoute:
    intent_mode = classify_intent(latest_user_message)
    if intent_mode == INTENT_SMALLTALK_META:
        return IntentRoute(
            mode=intent_mode, prompt=ai_prompts.SYSTEM_PROMPT_SMALLTALK_META, tools=[]
        )
    if intent_mode == INTENT_THEME_CHANGE:
        return IntentRoute(
            mode=intent_mode,
            prompt=ai_prompts.SYSTEM_PROMPT_THEME_CHANGE,
            tools=theme_tools,
        )
    return IntentRoute(mode=INTENT_SQL_AGENT, prompt=sql_prompt, tools=sql_tools)


def _latest_user_message_text(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _is_theme_tool(tool: Any) -> bool:
    return getattr(tool, "name", None) == "switch_color_theme"


class IntentRoutingMiddleware(AgentMiddleware[ChatAgentState, Any]):
    state_schema = ChatAgentState
    tools: list[Any] = []

    def __init__(self, sql_tools: list[Any], sql_prompt: str) -> None:
        self._theme_tools = [tool for tool in sql_tools if _is_theme_tool(tool)]
        self._sql_tools = [tool for tool in sql_tools if not _is_theme_tool(tool)]
        self._sql_prompt = sql_prompt

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        route = self._select_and_log_route(request)
        return handler(request.override(tools=route.tools, system_prompt=route.prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        route = self._select_and_log_route(request)
        return await handler(
            request.override(tools=route.tools, system_prompt=route.prompt)
        )

    def _select_and_log_route(self, request: ModelRequest) -> IntentRoute:
        route = select_intent_route(
            _latest_user_message_text(request),
            self._sql_tools,
            self._theme_tools,
            self._sql_prompt,
        )
        logger.debug(format_log_event("chat.intent", "selected", mode=route.mode))
        return route


__all__ = [
    "IntentRoute",
    "build_intent_routing_middleware",
    "select_intent_route",
]
