"""Compare baseline RAG retrieval vs agent reflection context recall."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from eval.agent.context import (
    GoldAgentContextCase,
    load_fixture_text,
    recall_on_context,
    recall_on_texts,
    run_crag_trim,
)
from eval.profiles import RAGProfile, build_retriever_for_profile
from eval.rag.metrics.recall import GoldRagCase, recall_at_k


class CompareCaseResult(TypedDict):
    case_id: str
    scenario: str
    pattern: str
    query: str
    top_k: int
    recall_baseline: float
    recall_agent: float
    delta: float
    recall_raw_fixture_at_1: NotRequired[float]
    notes: NotRequired[str]


class CompareSummary(TypedDict):
    dataset: str
    rag_profile: str
    collection: str
    top_k: int
    cases: int
    mean_recall_baseline: float
    mean_recall_agent: float
    mean_delta: float
    per_case: list[CompareCaseResult]


async def baseline_recall(
    case: GoldRagCase,
    profile: RAGProfile,
    collection: str,
    *,
    in_memory: bool,
    top_k: int,
) -> float:
    from eval.profiles import build_indexer_for_profile

    indexer = build_indexer_for_profile(profile, collection, in_memory=in_memory)
    retriever = build_retriever_for_profile(
        profile,
        collection,
        in_memory=in_memory,
        store=indexer.store,
        embedder=indexer.embedder,
    )
    try:
        chunks = await retriever.aquery(case["query"], top_k=top_k)
        return recall_at_k(chunks, case, k=top_k)
    finally:
        await indexer.store.aclose()


async def compare_case(
    case: GoldAgentContextCase,
    profile: RAGProfile,
    collection: str,
    *,
    in_memory: bool,
    top_k: int,
) -> CompareCaseResult:
    source = (case.get("expected_source") or "").strip()
    scenario = case["scenario"]

    recall_baseline = await baseline_recall(
        case,
        profile,
        collection,
        in_memory=in_memory,
        top_k=top_k,
    )

    if scenario == "direct_retrieve":
        recall_agent = recall_baseline
        return CompareCaseResult(
            case_id=case["case_id"],
            scenario=scenario,
            pattern=case["pattern"],
            query=case["query"],
            top_k=top_k,
            recall_baseline=recall_baseline,
            recall_agent=recall_agent,
            delta=recall_agent - recall_baseline,
            notes="react sanity：agent 臂与 baseline 共用 retriever",
        )

    if scenario == "crag_trim":
        noise = load_fixture_text(case["fixture_noise"])
        gold = load_fixture_text(case["fixture_gold"])
        labels = case.get("crag_labels") or []
        final_context = await run_crag_trim(case, noise=noise, gold=gold, labels=labels)
        recall_agent = recall_on_context(final_context, case, k=top_k, source=source)
        recall_raw_at_1 = recall_on_texts(
            [noise, gold],
            case,
            k=1,
            source=source,
        )
        return CompareCaseResult(
            case_id=case["case_id"],
            scenario=scenario,
            pattern=case["pattern"],
            query=case["query"],
            top_k=top_k,
            recall_baseline=recall_baseline,
            recall_agent=recall_agent,
            delta=recall_agent - recall_baseline,
            recall_raw_fixture_at_1=recall_raw_at_1,
            notes="fixture 噪声+gold；CRAG 保留 correct",
        )

    raise ValueError(f"Unknown agent context scenario {scenario!r}")


def summarize_compare(
    *,
    dataset: str,
    rag_profile: str,
    collection: str,
    top_k: int,
    results: list[CompareCaseResult],
) -> CompareSummary:
    if not results:
        return CompareSummary(
            dataset=dataset,
            rag_profile=rag_profile,
            collection=collection,
            top_k=top_k,
            cases=0,
            mean_recall_baseline=0.0,
            mean_recall_agent=0.0,
            mean_delta=0.0,
            per_case=[],
        )
    n = len(results)
    return CompareSummary(
        dataset=dataset,
        rag_profile=rag_profile,
        collection=collection,
        top_k=top_k,
        cases=n,
        mean_recall_baseline=sum(r["recall_baseline"] for r in results) / n,
        mean_recall_agent=sum(r["recall_agent"] for r in results) / n,
        mean_delta=sum(r["delta"] for r in results) / n,
        per_case=results,
    )
