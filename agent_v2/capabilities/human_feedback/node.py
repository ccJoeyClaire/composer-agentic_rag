"""Human feedback post-processor after ``request_clarification`` tool (stub)."""

from __future__ import annotations

from agent_v2.capabilities.human_feedback.config import HumanFeedbackConfig
from agent_v2.capabilities.human_feedback.metadata import (
    FEEDBACK_DECISION_KEY,
    FEEDBACK_QUESTION_KEY,
    FEEDBACK_TIMING_KEY,
)
from agent_v2.capabilities.human_feedback.tool import CLARIFICATION_TOOL_NAME
from agent_v2.core.edges.tool_calls import preceding_ai_tool_calls
from agent_v2.core.state import AgentState, merge_metadata
from agent_v2.types import FeedbackTiming


async def human_feedback_node(
    state: AgentState,
    *,
    capability_config: HumanFeedbackConfig,
) -> dict[str, object]:
    """Record clarification metadata after the LLM invoked ``request_clarification``.

    The turn ends at ``END`` — the user replies in a new graph invoke.
    Stub: extracts question/timing from the tool call args.
    """
    _ = capability_config
    tool_calls = preceding_ai_tool_calls(state)
    clarify_call = next(
        (c for c in tool_calls if c.get("name") == CLARIFICATION_TOOL_NAME),
        None,
    )
    if clarify_call is None:
        return merge_metadata(
            state,
            {FEEDBACK_DECISION_KEY: "finish"},
        )

    args = clarify_call.get("args") or {}
    timing_raw = str(args.get("timing", "pre_retrieval"))
    timing: FeedbackTiming = (
        "pre_answer" if timing_raw == "pre_answer" else "pre_retrieval"
    )

    return merge_metadata(
        state,
        {
            FEEDBACK_DECISION_KEY: "clarify",
            FEEDBACK_QUESTION_KEY: str(args.get("question", "")),
            FEEDBACK_TIMING_KEY: timing,
        },
    )
