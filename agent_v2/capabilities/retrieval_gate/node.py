"""Retrieval quality gate node (stub — verdict only, no actions)."""

from __future__ import annotations

from agent_v2.capabilities.retrieval_gate.config import RetrievalGateConfig
from agent_v2.capabilities.retrieval_gate.metadata import (
    GATE_ISSUES_KEY,
    GATE_PASSAGES_SUMMARY_KEY,
    GATE_VERDICT_KEY,
)
from agent_v2.core.state import AgentState, merge_metadata


async def retrieval_gate_node(
    state: AgentState,
    *,
    capability_config: RetrievalGateConfig,
) -> dict[str, object]:
    """Grade the latest RAG retrieval; LLM decides whether to retry or web-search.

    Stub: always passes until scoring logic is wired.
    """
    _ = capability_config
    return merge_metadata(
        state,
        {
            GATE_VERDICT_KEY: "pass",
            GATE_ISSUES_KEY: [],
            GATE_PASSAGES_SUMMARY_KEY: None,
        },
    )
