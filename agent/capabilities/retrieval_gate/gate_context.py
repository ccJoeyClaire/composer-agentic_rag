"""Transient retrieval-gate context for LLM calls (not persisted in state)."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage

from agent.capabilities.retrieval_gate.metadata import (
    GATE_ISSUES_KEY,
    GATE_PASSAGES_SUMMARY_KEY,
    GATE_VERDICT_KEY,
)
from agent.core.metadata.schema import AgentMetadata


def build_gate_context_message(meta: AgentMetadata) -> SystemMessage | None:
    """Build a system prompt from gate metadata, or None when not applicable.

    Scoring ``error`` verdicts are internal pipeline failures; they are not
    surfaced to the LLM (the gate node retries scoring silently).
    """
    verdict = meta.get(GATE_VERDICT_KEY)
    if verdict is None or verdict == "error":
        return None

    lines = [f"Retrieval gate verdict: {verdict}"]
    summary = meta.get(GATE_PASSAGES_SUMMARY_KEY)
    if summary:
        lines.append(f"Passages: {summary}")
    issues = meta.get(GATE_ISSUES_KEY) or []
    if issues:
        lines.append("Issues: " + "; ".join(issues))
    if verdict in ("low_quality", "empty"):
        lines.append(
            "Treat the current retrieval as insufficient — do not assume the passages "
            "are adequate evidence. Retry retrieval with a different query, adjust "
            "search parameters, or use web search when enabled. If you have exhausted "
            "reasonable retrieval and still cannot answer, you may give a final answer "
            "that begins with “证据不足” or “无法基于已有检索结果” (or the English equivalents)."
        )
    return SystemMessage(content="\n".join(lines))


def prepare_messages_for_llm(
    messages: list[BaseMessage],
    *,
    gate_message: SystemMessage | None,
) -> list[BaseMessage]:
    """Return a copy of *messages* with optional transient gate system text inserted."""
    if gate_message is None:
        return list(messages)

    insert_at = 0
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            insert_at = index + 1
        else:
            break

    result = list(messages)
    result.insert(insert_at, gate_message)
    return result
