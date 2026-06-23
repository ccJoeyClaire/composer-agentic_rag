"""Human feedback capability — injects a clarification tool with LangGraph interrupt."""

from __future__ import annotations

from agent_v2.capabilities.human_feedback.tool import (
    CLARIFICATION_TOOL_NAME,
    request_clarification,
)


class HumanFeedbackCapability:
    """Optional ``request_clarification`` tool; LLM decides when to call it."""

    name = "human_feedback"

    @staticmethod
    def extra_tools() -> dict[str, object]:
        return {CLARIFICATION_TOOL_NAME: request_clarification}
