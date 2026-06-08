"""Feedback subgraph — detect user correction and plan agent response (Phase 3)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import partial
from typing import Awaitable, Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from agent.metadata_schema import FeedbackKind, merge_metadata
from agent.state import AgentState
from agent.subgraph.Self_RAG import last_human_message
from llm.client import LLMClient

DetectFeedbackFn = Callable[[str], Awaitable[dict]]
PlanFeedbackFn = Callable[[str, dict], Awaitable[dict]]

FeedbackAction = Literal["none", "requery", "rewrite", "clarify"]


class FeedbackDetection(TypedDict, total=False):
    detected: bool
    kind: FeedbackKind | None


class FeedbackPlan(TypedDict, total=False):
    action: FeedbackAction
    kind: FeedbackKind | None
    suggested_query: str | None
    hint: str | None


CORRECTION_PATTERNS = (
    "wrong",
    "incorrect",
    "not right",
    "that's wrong",
    "that is wrong",
    "you're wrong",
    "you are wrong",
    "actually",
    "不对",
    "错了",
    "不正确",
    "不是这样的",
    "你说错",
    "纠正",
    "重新回答",
)
THUMBS_DOWN_PATTERNS = (
    "thumbs down",
    "thumb down",
    "👎",
    "不好",
    "不满意",
    "不行",
    "没用",
    "不靠谱",
)
CLARIFY_PATTERNS = (
    "what do you mean",
    "clarify",
    "不清楚",
    "什么意思",
    "能解释",
    "看不懂",
    "再说一遍",
    "详细点",
)


FEEDBACK_DETECT_PROMPT = """Detect whether the user message expresses feedback about a previous assistant answer.

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


def _match_patterns(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def rule_based_detect_feedback(text: str) -> FeedbackDetection:
    stripped = text.strip()
    if not stripped:
        return {"detected": False, "kind": None}

    if _match_patterns(stripped, CLARIFY_PATTERNS):
        return {"detected": True, "kind": "clarify"}
    if _match_patterns(stripped, THUMBS_DOWN_PATTERNS):
        return {"detected": True, "kind": "thumbs_down"}
    if _match_patterns(stripped, CORRECTION_PATTERNS):
        return {"detected": True, "kind": "correction"}

    return {"detected": False, "kind": None}


def _extract_query_candidate(text: str) -> str | None:
    quoted = re.findall(r"[\"'「『](.+?)[\"'」』]", text)
    if quoted:
        return quoted[-1].strip()
    cleaned = re.sub(
        r"(?i)(wrong|incorrect|不对|错了|不好|什么意思|what do you mean|actually)[,，:：\s]*",
        "",
        text,
    ).strip(" ?？.")
    return cleaned or None


def heuristic_plan_feedback(text: str, metadata: dict) -> FeedbackPlan:
    detection = rule_based_detect_feedback(text)
    kind = detection.get("kind")
    prior_query = str(metadata.get("rag_last_query") or "")
    had_retrieval = bool(metadata.get("rag_last_raw"))

    if not detection.get("detected"):
        return {"action": "none", "kind": None, "suggested_query": None, "hint": None}

    if kind == "clarify":
        return {
            "action": "clarify",
            "kind": kind,
            "suggested_query": None,
            "hint": "Ask the user a clarifying question about what was unclear.",
        }

    suggested = _extract_query_candidate(text) or prior_query or text
    if kind in {"correction", "thumbs_down"} and had_retrieval:
        return {
            "action": "requery",
            "kind": kind,
            "suggested_query": suggested,
            "hint": (
                "User rejected the previous answer. Search again with a refined query "
                f"before responding: {suggested}"
            ),
        }

    return {
        "action": "rewrite",
        "kind": kind,
        "suggested_query": None,
        "hint": "User rejected the previous answer. Rewrite the response carefully.",
    }


async def default_detect_feedback(llm: LLMClient, text: str) -> FeedbackDetection:
    response = await llm.arequest_llm(
        [
            {
                "role": "user",
                "content": FEEDBACK_DETECT_PROMPT.format(message=text),
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
    text: str,
    metadata: dict,
) -> FeedbackPlan:
    response = await llm.arequest_llm(
        [
            {
                "role": "user",
                "content": FEEDBACK_PLAN_PROMPT.format(
                    message=text,
                    prior_query=metadata.get("rag_last_query") or "(none)",
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
        kind = rule_based_detect_feedback(text).get("kind")

    return {
        "action": action,
        "kind": kind,
        "suggested_query": payload.get("suggested_query"),
        "hint": payload.get("hint"),
    }


def _clear_feedback_metadata() -> dict:
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

    text = str(human.content or "")
    if detect_fn is not None:
        detection = await detect_fn(text)
    elif llm is not None:
        detection = await default_detect_feedback(llm, text)
    else:
        detection = rule_based_detect_feedback(text)

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
    meta = state.get("metadata") or {}
    if meta.get("feedback_detected"):
        return "plan_feedback"
    return "feedback_exit"


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
    text = str(human.content or "") if human else ""

    if plan_fn is not None:
        plan = await plan_fn(text, meta)
    elif llm is not None:
        plan = await default_plan_feedback(llm, text, meta)
    else:
        plan = heuristic_plan_feedback(text, meta)

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


async def feedback_exit_node(state: AgentState) -> dict:
    return {}


def build_feedback_subgraph(config: FeedbackConfig):
    graph = StateGraph(AgentState)

    graph.add_node(
        "detect_feedback",
        partial(
            detect_feedback_node,
            llm=config.llm,
            detect_fn=config.detect_fn,
        ),
    )
    graph.add_node(
        "plan_feedback",
        partial(
            plan_feedback_node,
            llm=config.llm,
            plan_fn=config.plan_fn,
        ),
    )
    graph.add_node("feedback_exit", feedback_exit_node)

    graph.set_entry_point("detect_feedback")
    graph.add_conditional_edges(
        "detect_feedback",
        route_after_detect,
        {"plan_feedback": "plan_feedback", "feedback_exit": "feedback_exit"},
    )
    graph.add_edge("plan_feedback", "feedback_exit")
    graph.add_edge("feedback_exit", END)

    return graph.compile()
