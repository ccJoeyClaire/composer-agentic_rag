"""Clarification tool — LLM calls this to pause for human input via LangGraph interrupt."""

from __future__ import annotations

from langgraph.types import interrupt

CLARIFICATION_TOOL_NAME = "request_clarification"


def request_clarification(question: str, timing: str = "pre_retrieval") -> str:
    """Pause the graph and ask the user for clarification.

    Resume with ``Command(resume=...)``; the value becomes the tool result.
    """
    user_answer = interrupt(
        {
            "type": "clarification",
            "question": question,
            "timing": timing,
        }
    )
    return str(user_answer)
