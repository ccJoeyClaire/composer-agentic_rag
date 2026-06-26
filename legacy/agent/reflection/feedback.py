"""Feedback reflection nodes — detect user correction and plan agent response.

Plain graph nodes composed directly in ``agent/graph.py``. ``route_after_detect``
returns the next routing token ("plan_feedback" to plan, "continue" to skip).

Detection and planning require an :class:`LLMClient` (or injected ``detect_fn`` /
``plan_fn``). There is no rule-based fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable, TypedDict

from langchain_core.messages import HumanMessage

from legacy.agent.reflection.self_rag import last_human_message
from legacy.agent.state import (
    AgentMetadata,
    AgentState,
    FeedbackAction,
    FeedbackKind,
    merge_metadata,
)
from llm.client import LLMClient


class FeedbackDetection(TypedDict, total=False):
    detected: bool
    kind: FeedbackKind | None


class FeedbackPlan(TypedDict, total=False):
    action: FeedbackAction
    kind: FeedbackKind | None
    suggested_query: str | None
    hint: str | None


class FeedbackMetadataFields(TypedDict):
    feedback_detected: bool
    feedback_kind: FeedbackKind | None
    feedback_action: FeedbackAction
    feedback_suggested_query: str | None
    feedback_hint: str | None


DetectFeedbackFn = Callable[[str], Awaitable[FeedbackDetection]]
PlanFeedbackFn = Callable[[str, AgentMetadata], Awaitable[FeedbackPlan]]


FEEDBACK_DETECT_PROMPT = """Detect whether the user message expresses feedback about a previous assistant answer.

Kinds:
- correction: user says the answer is wrong or factually incorrect
- thumbs_down: user dislikes or rejects the answer without giving a specific fix
- clarify: user did not understand the answer and wants explanation

User message:
{message}

Return JSON only:
{{"detected": true|false, "kind": "correction"|"thumbs_down"|"clarify"|null}}
"""

FEEDBACK_PLAN_PROMPT = """Plan how the assistant should respond to user feedback.

User feedback:
{message}

Previous retrieval query (if any):
{prior_query}

Had retrieval context: {had_retrieval}

Return JSON only:
{{
  "action": "requery"|"rewrite"|"clarify",
  "kind": "correction"|"thumbs_down"|"clarify",
  "suggested_query": "search query string or null",
  "hint": "short instruction for the assistant"
}}
"""


@dataclass
class FeedbackConfig:
    llm: LLMClient | None = None
    detect_fn: DetectFeedbackFn | None = None
    plan_fn: PlanFeedbackFn | None = None


async def default_detect_feedback(
    llm: LLMClient,
    user_feedback_message: str,
) -> FeedbackDetection:
    response = await llm.arequest_llm(
        [
            {
                "role": "user",
                "content": FEEDBACK_DETECT_PROMPT.format(message=user_feedback_message),
            }
        ],
        json_output=True,
    )
    try:
        payload = json.loads(response.content or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not payload.get("detected"):
        return {"detected": False, "kind": None}
    kind = payload.get("kind")
    if kind not in {"correction", "thumbs_down", "clarify"}:
        kind = None
    return {"detected": True, "kind": kind}


async def default_plan_feedback(
    llm: LLMClient,
    user_feedback_message: str,
    metadata: AgentMetadata,
) -> FeedbackPlan:
    response = await llm.arequest_llm(
        [
            {
                "role": "user",
                "content": FEEDBACK_PLAN_PROMPT.format(
                    message=user_feedback_message,
                    prior_query=metadata.get("rag_last_query") or "(none)",
                    had_retrieval=bool(metadata.get("rag_last_raw")),
                ),
            }
        ],
        json_output=True,
    )
    try:
        payload = json.loads(response.content or "{}")
    except json.JSONDecodeError:
        payload = {}

    action = payload.get("action", "rewrite")
    if action not in {"requery", "rewrite", "clarify"}:
        action = "rewrite"
    kind = payload.get("kind")
    if kind not in {"correction", "thumbs_down", "clarify"}:
        kind = metadata.get("feedback_kind")

    return {
        "action": action,
        "kind": kind,
        "suggested_query": payload.get("suggested_query"),
        "hint": payload.get("hint"),
    }


def _clear_feedback_metadata() -> FeedbackMetadataFields:
    return {
        "feedback_detected": False,
        "feedback_kind": None,
        "feedback_action": "none",
        "feedback_suggested_query": None,
        "feedback_hint": None,
    }


async def detect_feedback_node(
    state: AgentState,
    *,
    llm: LLMClient | None,
    detect_fn: DetectFeedbackFn | None,
) -> dict:
    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
        return merge_metadata(state, _clear_feedback_metadata())

    human = last_human_message(state["messages"])
    if human is None:
        return merge_metadata(state, _clear_feedback_metadata())

    user_feedback_message = str(human.content or "")
    if not user_feedback_message.strip():
        return merge_metadata(state, _clear_feedback_metadata())

    if detect_fn is not None:
        detection = await detect_fn(user_feedback_message)
    elif llm is not None:
        detection = await default_detect_feedback(llm, user_feedback_message)
    else:
        return merge_metadata(state, _clear_feedback_metadata())

    if not detection.get("detected"):
        return merge_metadata(state, _clear_feedback_metadata())

    return merge_metadata(
        state,
        {
            "feedback_detected": True,
            "feedback_kind": detection.get("kind"),
        },
    )


def route_after_detect(state: AgentState) -> str:
    """Return "plan_feedback" when feedback detected, else "continue"."""
    meta = state.get("metadata") or {}
    if meta.get("feedback_detected"):
        return "plan_feedback"
    return "continue"


async def plan_feedback_node(
    state: AgentState,
    *,
    llm: LLMClient | None,
    plan_fn: PlanFeedbackFn | None,
) -> dict:
    meta = state.get("metadata") or {}
    if not meta.get("feedback_detected"):
        return merge_metadata(state, _clear_feedback_metadata())

    human = last_human_message(state["messages"])
    user_feedback_message = str(human.content or "") if human else ""

    if plan_fn is not None:
        plan = await plan_fn(user_feedback_message, meta)
    elif llm is not None:
        plan = await default_plan_feedback(llm, user_feedback_message, meta)
    else:
        return merge_metadata(
            state,
            {
                "feedback_detected": True,
                "feedback_kind": meta.get("feedback_kind"),
                "feedback_action": "rewrite",
                "feedback_suggested_query": None,
                "feedback_hint": None,
            },
        )

    action = plan.get("action", "none")
    if action not in {"requery", "rewrite", "clarify", "none"}:
        action = "none"

    return merge_metadata(
        state,
        {
            "feedback_detected": True,
            "feedback_kind": plan.get("kind") or meta.get("feedback_kind"),
            "feedback_action": action,
            "feedback_suggested_query": plan.get("suggested_query"),
            "feedback_hint": plan.get("hint"),
        },
    )
