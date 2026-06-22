"""Clarification tool — LLM calls this when it wants human input."""

from __future__ import annotations

CLARIFICATION_TOOL_NAME = "request_clarification"


def request_clarification(
    question: str,
    timing: str = "pre_retrieval",
) -> str:
    """Ask the user for clarification before continuing.

    Args:
        question: The clarification question to show the user.
        timing: ``pre_retrieval`` or ``pre_answer`` — when in the flow the
            LLM decided clarification is needed.

    Returns:
        Acknowledgement string (actual UX handled by ``human_feedback_node``).
    """
    return f"clarification_requested:{timing}:{question}"
