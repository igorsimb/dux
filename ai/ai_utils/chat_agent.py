"""Shared chat agent builder backed by LangChain create_agent."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from ai import ai_prompts

from .agent_state import ChatAgentState
from .checkpointer import get_checkpointer
from .intent_middleware import build_intent_routing_middleware
from .memory_middleware import trim_chat_history_middleware


def build_chat_agent(model: Any, tools: list[Any]):
    """Build the guarded chat agent used by the web view.

    The SQL system prompt (SYSTEM_PROMPT_RU_SARCASTIC) is the default prompt
    passed to `create_agent(...)`. Chat history trimming runs first, then
    human-in-the-loop protection can pause unsafe tool calls, and finally
    intent-routing middleware can swap the prompt and available tools for
    `smalltalk_meta` or `theme_change` turns.
    """

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC,
        middleware=[
            trim_chat_history_middleware,
            HumanInTheLoopMiddleware(
                interrupt_on={"ask_user": {"allowed_decisions": ["respond"]}},
            ),
            build_intent_routing_middleware(
                tools, ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC
            ),
        ],
        state_schema=ChatAgentState,
        checkpointer=get_checkpointer(),
    )
