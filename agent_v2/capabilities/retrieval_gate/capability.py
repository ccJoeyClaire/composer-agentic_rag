"""Retrieval gate capability registration."""

from __future__ import annotations

from functools import partial

from langgraph.graph import StateGraph

from agent_v2.capabilities.retrieval_gate.config import RetrievalGateConfig
from agent_v2.capabilities.retrieval_gate.node import retrieval_gate_node
from agent_v2.config import AgentConfig
from agent_v2.core.edges.names import NodeName


class RetrievalGateCapability:
    """Post-RAG quality gate — reports issues; does not auto-requery or web-search."""

    def register(self, graph: StateGraph, config: AgentConfig) -> None:
        capability_config = config.retrieval_gate or RetrievalGateConfig()
        graph.add_node(
            NodeName.RETRIEVAL_GATE,
            partial(
                retrieval_gate_node,
                capability_config=capability_config,
                rag_tool_name=config.rag_tool_name,
            ),
        )
