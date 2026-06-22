"""Helpers for inspecting tool calls on the message history."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent_v2.core.constants import DEFAULT_RAG_TOOL_NAME
from agent_v2.core.state import AgentState


def last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Return the most recent AIMessage, or None."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def tool_calls_from_ai(ai_msg: AIMessage) -> list[dict[str, Any]]:
    """Normalize ``tool_calls`` on an AIMessage to a list (empty when absent)."""
    return list(getattr(ai_msg, "tool_calls", None) or [])


def preceding_ai_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """Tool calls from the AIMessage that triggered the latest tool batch."""
    messages = state["messages"]
    if not messages:
        return []

    if isinstance(messages[-1], ToolMessage):
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                return tool_calls_from_ai(msg)
        return []

    last = messages[-1]
    if isinstance(last, AIMessage):
        return tool_calls_from_ai(last)
    return []


def has_tool_named(tool_calls: list[dict[str, Any]], name: str) -> bool:
    """True when any tool call targets *name*."""
    return any(call.get("name") == name for call in tool_calls)


def has_rag_tool_call(
    tool_calls: list[dict[str, Any]],
    *,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> bool:
    """True when the batch includes a RAG search tool call."""
    return has_tool_named(tool_calls, rag_tool_name)


def last_batch_included_rag(
    state: AgentState,
    *,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> bool:
    """True when the tool batch just executed included RAG."""
    return has_rag_tool_call(
        preceding_ai_tool_calls(state),
        rag_tool_name=rag_tool_name,
    )
