"""Rag profile router capability registration."""

from __future__ import annotations

from functools import partial

from langgraph.graph import StateGraph

from agent_v2.capabilities.rag_profile_router.config import RagProfileRouterConfig
from agent_v2.capabilities.rag_profile_router.node import rag_profile_router_node
from agent_v2.config import AgentConfig
from agent_v2.core.edges.names import NodeName


class RagProfileRouterCapability:
    """Validate LLM RAG profile overrides before tool execution; enables override schema."""

    def register(self, graph: StateGraph, config: AgentConfig) -> None:
        capability_config = config.rag_profile_router or RagProfileRouterConfig()
        graph.add_node(
            NodeName.RAG_PROFILE_ROUTER,
            partial(
                rag_profile_router_node,
                agent_config=config,
                capability_config=capability_config,
            ),
        )
