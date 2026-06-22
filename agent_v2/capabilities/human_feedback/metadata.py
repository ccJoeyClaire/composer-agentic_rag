"""Metadata written by the human_feedback capability."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from agent_v2.types import FeedbackDecision, FeedbackTiming

FEEDBACK_DECISION_KEY = "feedback_decision"
FEEDBACK_QUESTION_KEY = "feedback_question"
FEEDBACK_TIMING_KEY = "feedback_timing"


class HumanFeedbackMeta(TypedDict, total=False):
    """Record of an LLM-initiated clarification request."""

    feedback_decision: FeedbackDecision
    feedback_question: str | None
    feedback_timing: FeedbackTiming
