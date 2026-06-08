"""Agent reflection metadata fields (CRAG / Self-RAG / Feedback)."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from agent.state import AgentState

CragVerdict = Literal["correct", "incorrect", "ambiguous", "skipped"]
CragAction = Literal["use", "requery", "web_fallback", "degrade"]
FeedbackKind = Literal["correction", "thumbs_down", "clarify"]
FeedbackAction = Literal["none", "requery", "rewrite", "clarify"]


class CragLabel(TypedDict):
    index: int
    label: CragVerdict
    score: NotRequired[float]


class AgentMetadata(TypedDict, total=False):
    # RAG invocation
    rag_tool_name: str
    rag_attempt: int
    rag_last_query: str
    rag_last_raw: str | None
    max_rag_attempts: int

    # CRAG (#17)
    crag_verdict: CragVerdict
    crag_labels: list[CragLabel]
    crag_action: CragAction
    crag_requery_hint: str | None

    # Self-RAG (#12)
    self_rag_need_retrieve: bool | None
    self_rag_grounded: bool | None
    self_rag_retry_allowed: bool
    self_rag_retrieve_hint: str | None
    self_rag_retry_hint: str | None

    # Feedback (#11)
    feedback_detected: bool
    feedback_kind: FeedbackKind | None
    feedback_action: FeedbackAction
    feedback_suggested_query: str | None
    feedback_hint: str | None


DEFAULT_RAG_TOOL_NAME = "RAG_search_tool"
DEFAULT_MAX_RAG_ATTEMPTS = 2


def get_metadata(state: AgentState) -> dict[str, Any]:
    return dict(state.get("metadata") or {})


def merge_metadata(state: AgentState, patch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    current = get_metadata(state)
    current.update(patch)
    return {"metadata": current}
