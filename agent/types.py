"""Shared literals and enums for agent."""

from __future__ import annotations

from typing import Literal

GateVerdict = Literal["pass", "low_quality", "empty", "error"]

FeedbackDecision = Literal["clarify", "continue", "finish"]
FeedbackTiming = Literal["pre_retrieval", "pre_answer"]
