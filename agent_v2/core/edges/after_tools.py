"""Routing after the tools node — split RAG vs web/other paths."""

from __future__ import annotations

from agent_v2.capabilities.human_feedback.tool import CLARIFICATION_TOOL_NAME
from agent_v2.config import AgentConfig
from agent_v2.core.edges.names import NodeName
from agent_v2.core.edges.tool_calls import (
    has_tool_named,
    last_batch_included_rag,
    preceding_ai_tool_calls,
)
from agent_v2.core.state import AgentState


def route_after_tools(state: AgentState, *, agent_config: AgentConfig) -> str:
    """Route from ``tools`` back toward ``llm`` or capability post-processors.

    - Clarification tool (when human_feedback enabled) → ``human_feedback`` → END
    - RAG tool batch + gate enabled → ``retrieval_gate`` → ``llm``
    - Everything else → ``llm``
    """
    tool_calls = preceding_ai_tool_calls(state)

    if agent_config.enable_human_feedback and has_tool_named(
        tool_calls,
        CLARIFICATION_TOOL_NAME,
    ):
        return NodeName.HUMAN_FEEDBACK

    if agent_config.enable_retrieval_gate and last_batch_included_rag(
        state,
        rag_tool_name=agent_config.rag_tool_name,
    ):
        return NodeName.RETRIEVAL_GATE

    return NodeName.LLM
