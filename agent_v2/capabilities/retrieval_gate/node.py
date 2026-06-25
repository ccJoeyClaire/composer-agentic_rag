"""Retrieval quality gate node — score passages and write verdict metadata."""

from __future__ import annotations

from agent_v2.capabilities.retrieval_gate.config import RetrievalGateConfig
from agent_v2.capabilities.retrieval_gate.metadata import (
    GATE_ISSUES_KEY,
    GATE_PASSAGES_SUMMARY_KEY,
    GATE_VERDICT_KEY,
)
from agent_v2.capabilities.retrieval_gate.rag_context import (
    extract_latest_rag_context,
    split_rag_passages,
)
from agent_v2.capabilities.retrieval_gate.score import build_dashscope_score_fn
from agent_v2.capabilities.retrieval_gate.verdict import (
    build_passages_summary,
    compute_gate_verdict,
)
from agent_v2.core.constants import DEFAULT_RAG_TOOL_NAME
from agent_v2.core.state import AgentState, merge_metadata


def _resolve_score_fn(config: RetrievalGateConfig):
    if config.score_fn is not None:
        return config.score_fn
    return build_dashscope_score_fn(config.rerank_client)


async def retrieval_gate_node(
    state: AgentState,
    *,
    capability_config: RetrievalGateConfig,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> dict[str, object]:
    """Grade the latest RAG retrieval; LLM decides whether to retry or web-search."""
    context = extract_latest_rag_context(
        state["messages"],
        rag_tool_name=rag_tool_name,
    )
    if context is None:
        return merge_metadata(
            state,
            {
                GATE_VERDICT_KEY: "empty",
                GATE_ISSUES_KEY: ["no RAG tool result in the latest batch"],
                GATE_PASSAGES_SUMMARY_KEY: None,
            },
        )

    query, raw = context
    passages = split_rag_passages(raw)
    if not passages:
        return merge_metadata(
            state,
            {
                GATE_VERDICT_KEY: "empty",
                GATE_ISSUES_KEY: ["RAG tool returned no passages"],
                GATE_PASSAGES_SUMMARY_KEY: build_passages_summary([], []),
            },
        )

    score_fn = _resolve_score_fn(capability_config)
    scores = await score_fn(state, query, passages)
    verdict, issues = compute_gate_verdict(
        passages,
        scores,
        pass_threshold=capability_config.pass_threshold,
    )
    return merge_metadata(
        state,
        {
            GATE_VERDICT_KEY: verdict,
            GATE_ISSUES_KEY: issues,
            GATE_PASSAGES_SUMMARY_KEY: build_passages_summary(passages, scores),
        },
    )
