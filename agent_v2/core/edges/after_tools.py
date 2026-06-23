"""Routing after the tools node — split RAG vs web/other paths."""

from __future__ import annotations

from agent_v2.config import AgentConfig
from agent_v2.core.edges.names import NodeName
from agent_v2.core.edges.tool_calls import last_batch_included_rag
from agent_v2.core.state import AgentState


def route_after_tools(state: AgentState, *, agent_config: AgentConfig) -> str:
    """Route from ``tools`` back toward ``llm`` or ``retrieval_gate``.

    Clarification uses ``interrupt()`` inside ``request_clarification``; no extra node.
    """
    if agent_config.enable_retrieval_gate and last_batch_included_rag(
        state,
        rag_tool_name=agent_config.rag_tool_name,
    ):
        return NodeName.RETRIEVAL_GATE

    return NodeName.LLM
