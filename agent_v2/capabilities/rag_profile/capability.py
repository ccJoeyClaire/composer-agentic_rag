"""Rag profile capability registration."""

from __future__ import annotations

from functools import partial

from langgraph.graph import StateGraph

from agent_v2.capabilities.rag_profile.config import RagProfileConfig
from agent_v2.capabilities.rag_profile.node import rag_profile_router_node
from agent_v2.config import AgentConfig
from agent_v2.core.edges.names import NodeName


class RagProfileCapability:
    """Validate and record LLM-specified RAG profiles before tool execution."""

    name = "rag_profile"

    def register(self, graph: StateGraph, config: AgentConfig) -> None:
        capability_config = config.rag_profile or RagProfileConfig()
        graph.add_node(
            NodeName.RAG_PROFILE_ROUTER,
            partial(
                rag_profile_router_node,
                agent_config=config,
                capability_config=capability_config,
            ),
        )
