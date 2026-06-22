"""Metadata written by the retrieval_gate capability (verdict only — no actions)."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from agent_v2.types import GateVerdict

GATE_VERDICT_KEY = "gate_verdict"
GATE_ISSUES_KEY = "gate_issues"
GATE_PASSAGES_SUMMARY_KEY = "gate_passages_summary"


class RetrievalGateMeta(TypedDict, total=False):
    """Quality assessment of the latest RAG retrieval — LLM decides what to do next."""

    gate_verdict: GateVerdict
    gate_issues: list[str]
    gate_passages_summary: str | None
