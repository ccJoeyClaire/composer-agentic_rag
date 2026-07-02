"""Retrieval quality gate node — score passages and write verdict metadata."""

from __future__ import annotations

from agent.capabilities.retrieval_gate.config import RetrievalGateConfig
from agent.capabilities.retrieval_gate.evidence import extract_latest_evidence_batch
from agent.capabilities.retrieval_gate.metadata import (
    GATE_EVIDENCE_SOURCES_KEY,
    GATE_ISSUES_KEY,
    GATE_PASSAGES_SUMMARY_KEY,
    GATE_VERDICT_KEY,
)
from agent.capabilities.retrieval_gate.score import build_dashscope_score_fn
from agent.capabilities.retrieval_gate.verdict import (
    build_passages_summary,
    compute_gate_verdict,
)
from agent.core.tool_box import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME
from agent.core.state import AgentState, merge_metadata


def _resolve_score_fn(config: RetrievalGateConfig):
    if config.score_fn is not None:
        return config.score_fn
    return build_dashscope_score_fn(config.rerank_client)


async def retrieval_gate_node(
    state: AgentState,
    *,
    capability_config: RetrievalGateConfig,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
    web_tool_name: str = DEFAULT_WEB_TOOL_NAME,
) -> dict[str, object]:
    """Grade the latest RAG/web retrieval; LLM decides whether to retry or answer."""
    batch = extract_latest_evidence_batch(
        state["messages"],
        rag_tool_name=rag_tool_name,
        web_tool_name=web_tool_name,
    )
    if batch is None:
        return merge_metadata(
            state,
            {
                GATE_VERDICT_KEY: "empty",
                GATE_ISSUES_KEY: ["no scorable tool result in the latest batch"],
                GATE_PASSAGES_SUMMARY_KEY: None,
                GATE_EVIDENCE_SOURCES_KEY: [],
            },
        )

    passages = batch.passages
    if not passages:
        return merge_metadata(
            state,
            {
                GATE_VERDICT_KEY: "empty",
                GATE_ISSUES_KEY: ["retrieval tools returned no passages"],
                GATE_PASSAGES_SUMMARY_KEY: build_passages_summary([], []),
                GATE_EVIDENCE_SOURCES_KEY: list(batch.sources),
            },
        )

    score_fn = _resolve_score_fn(capability_config)
    scores: list[float] = []
    verdict: str = "error"
    issues: list[str] = []

    for _ in range(capability_config.max_scoring_retries):
        scores = await score_fn(state, batch.user_query, passages)
        verdict, issues = compute_gate_verdict(
            passages,
            scores,
            pass_threshold=capability_config.pass_threshold,
        )
        if verdict != "error":
            break

    patch = merge_metadata(
        state,
        {
            GATE_VERDICT_KEY: verdict,
            GATE_ISSUES_KEY: issues,
            GATE_PASSAGES_SUMMARY_KEY: build_passages_summary(passages, scores),
            GATE_EVIDENCE_SOURCES_KEY: list(batch.sources),
        },
    )
    if verdict == "error":
        patch["error"] = issues[0] if issues else "retrieval gate scoring failed"
    return patch
