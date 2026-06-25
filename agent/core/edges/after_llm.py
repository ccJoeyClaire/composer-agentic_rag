"""Routing after the LLM node — LLM is the decision hub."""

from __future__ import annotations

from langgraph.graph import END

from agent.config import AgentConfig
from agent.core.edges.gate_policy import gate_blocks_final_answer
from agent.core.edges.names import NodeName
from agent.core.edges.tool_calls import (
    has_rag_tool_call,
    last_ai_message,
    tool_calls_from_ai,
)
from agent.core.state import AgentState


def route_after_llm(state: AgentState, *, agent_config: AgentConfig) -> str:
    """Route from ``llm`` based on the latest AIMessage.

    - RAG tool calls + profile router enabled → ``rag_profile_router``
    - Other tool calls → ``tools``
    - No tool calls → ``END`` (LLM decides the turn is done; no forced feedback)
    """
    ai_msg = last_ai_message(state["messages"])
    if ai_msg is None:
        return END

    tool_calls = tool_calls_from_ai(ai_msg)
    if not tool_calls:
        if gate_blocks_final_answer(state, agent_config=agent_config):
            return NodeName.GATE_REJECT
        return END

    if agent_config.enable_rag_profile_router and has_rag_tool_call(
        tool_calls,
        rag_tool_name=agent_config.rag_tool_name,
    ):
        return NodeName.RAG_PROFILE_ROUTER

    return NodeName.TOOLS
