"""Human feedback capability registration."""

from __future__ import annotations

from functools import partial

from langgraph.graph import StateGraph

from agent_v2.capabilities.human_feedback.config import HumanFeedbackConfig
from agent_v2.capabilities.human_feedback.node import human_feedback_node
from agent_v2.capabilities.human_feedback.tool import (
    CLARIFICATION_TOOL_NAME,
    request_clarification,
)
from agent_v2.config import AgentConfig
from agent_v2.core.edges.names import NodeName


class HumanFeedbackCapability:
    """Optional clarification tool + post-processor — LLM decides when to use it."""

    name = "human_feedback"

    def register(self, graph: StateGraph, config: AgentConfig) -> None:
        capability_config = config.human_feedback or HumanFeedbackConfig(llm=config.llm)
        graph.add_node(
            NodeName.HUMAN_FEEDBACK,
            partial(
                human_feedback_node,
                capability_config=capability_config,
            ),
        )

    @staticmethod
    def extra_tools() -> dict[str, object]:
        """Tools injected into :class:`AgentToolBox` when this capability is enabled."""
        return {CLARIFICATION_TOOL_NAME: request_clarification}
