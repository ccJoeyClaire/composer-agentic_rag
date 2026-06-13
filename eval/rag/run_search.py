"""
Run gold RAG queries against an indexed collection.

Usage (repo root):
  python -m eval.rag.run_search --dataset smoke --profile baseline
  python -m eval.rag.run_search --dataset smoke --profile baseline --trace
  python -m eval.rag.run_search --dataset smoke --profile baseline --trace --out eval/results
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NotRequired, TypedDict

from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.rag.metrics.recall import GoldRagCase, chunk_matches, load_gold_cases, mean_recall_at_k, recall_at_k
from eval.paths import dataset_dir
from eval.profiles import RAGProfile, build_retriever_for_profile, collection_name
from rag.base import (
    TRACE_HYDE_DOCUMENT_KEY,
    TRACE_RERANKED_KEY,
    TRACE_RETRIEVED_KEY,
    TRACE_WORKING_QUERY_KEY,
    Chunk,
)
from rag.config import get_rag_config

# ---------------------------------------------------------------------------
# Per-query trace types
# ---------------------------------------------------------------------------

CONTENT_PREVIEW_CHARS = 200


class StageChunk(TypedDict):
    """A single chunk's snapshot at one pipeline stage.

    ``rank`` is 0-based position in the stage's list.
    ``is_gold`` marks whether this chunk satisfies the gold case's criteria.
    """

    chunk_id: NotRequired[str]
    source: NotRequired[str]
    heading_path: NotRequired[str]
    score: float
    rank: int
    is_gold: bool
    content_preview: str


class QueryTrace(TypedDict):
    """Full per-query diagnostic record, one line in ``traces.jsonl``."""

    query: str
    working_query: str          # effective query after transform (== query if no transformer)
    hyde_document: NotRequired[str]
    retrieved: list[StageChunk]  # fetch_k results straight from vector store
    reranked: NotRequired[list[StageChunk]]   # rerank order, only present when reranker is on
    final: list[StageChunk]     # top_k results fed to downstream LLM
    recall_at_fetch_k: float    # gold hit anywhere in the *retrieved* list
    recall_at_top_k: float      # gold hit in the final top_k list


class SearchCaseResult(TypedDict):
    query: str
    recall_at_k: float
    top_preview: list[str]


class SearchDatasetSummary(TypedDict):
    """Aggregate run summary written to ``summary.json``."""

    dataset: str
    profile: str
    collection: str
    top_k: int
    fetch_k: int
    cases: int
    mean_recall_at_top_k: float
    mean_recall_at_fetch_k: float
    timestamp_utc: str
    per_case: list[SearchCaseResult]  # lightweight; full traces go to traces.jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_stage_chunks(
    chunks: list[Chunk],
    case: GoldRagCase,
) -> list[StageChunk]:
    result: list[StageChunk] = []
    for rank, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        sc = StageChunk(
            score=chunk.score,
            rank=rank,
            is_gold=chunk_matches(case, chunk.content, dict(meta)),
            content_preview=chunk.content[:CONTENT_PREVIEW_CHARS],
        )
        chunk_id = meta.get("chunk_id")
        if chunk_id:
            sc["chunk_id"] = chunk_id
        source = meta.get("source")
        if source:
            sc["source"] = source
        heading_path = meta.get("heading_path")
        if heading_path:
            sc["heading_path"] = heading_path
        result.append(sc)
    return result


def _recall_in_stage(stage_chunks: list[StageChunk]) -> float:
    return 1.0 if any(sc["is_gold"] for sc in stage_chunks) else 0.0


# ---------------------------------------------------------------------------
# Core async functions
# ---------------------------------------------------------------------------


async def trace_dataset(
    *,
    dataset: str,
    profile: RAGProfile,
    in_memory: bool,
    top_k: int | None = None,
    recall_n: int | None = None,
    collection: str | None = None,
) -> tuple[SearchDatasetSummary, list[QueryTrace]]:
    """Run every gold case through ``aquery_trace`` and collect full traces.

    Args:
        dataset: Name of the eval dataset (``smoke`` or a datasets/ subdirectory).
        profile: The ``RAGProfile`` to evaluate.
        in_memory: Use in-memory Qdrant instead of a persistent instance.
        top_k: Override config ``top_k``.
        recall_n: Override config ``recall_n``.
        collection: Override collection name.

    Returns:
        A ``(summary, traces)`` pair; write summary to ``summary.json`` and
        traces to ``traces.jsonl`` (one JSON object per line).
    """
    retriever_cfg = get_rag_config().retriever
    resolved_top_k = top_k if top_k is not None else retriever_cfg.top_k
    resolved_recall_n = recall_n if recall_n is not None else retriever_cfg.recall_n
    coll = collection or collection_name(dataset, profile.profile_id)

    from eval.profiles import build_indexer_for_profile

    indexer = build_indexer_for_profile(profile, coll, in_memory=in_memory)
    retriever = build_retriever_for_profile(
        profile,
        coll,
        in_memory=in_memory,
        store=indexer.store,
        embedder=indexer.embedder,
        recall_n=resolved_recall_n,
    )
    # fetch_k mirrors the logic inside RAGRetriever._run_query
    fetch_k = max(resolved_recall_n, resolved_top_k) if profile.use_reranker else resolved_top_k

    gold_path = dataset_dir(dataset) / "gold_rag.jsonl"
    cases = load_gold_cases(gold_path)

    per_case_summary: list[SearchCaseResult] = []
    traces: list[QueryTrace] = []
    top_k_scores: list[float] = []
    fetch_k_scores: list[float] = []

    for case in cases:
        result = await retriever.aquery_trace(case["query"], top_k=resolved_top_k)
        meta = result.metadata

        retrieved_chunks: list[Chunk] = meta.get(TRACE_RETRIEVED_KEY) or []
        reranked_chunks: list[Chunk] | None = meta.get(TRACE_RERANKED_KEY)
        final_chunks = result.chunks

        retrieved_stage = _to_stage_chunks(retrieved_chunks, case)
        final_stage = _to_stage_chunks(final_chunks, case)

        recall_fetch = _recall_in_stage(retrieved_stage)
        recall_top = recall_at_k(final_chunks, case, k=resolved_top_k)

        fetch_k_scores.append(recall_fetch)
        top_k_scores.append(recall_top)

        trace = QueryTrace(
            query=case["query"],
            working_query=meta.get(TRACE_WORKING_QUERY_KEY) or case["query"],
            retrieved=retrieved_stage,
            final=final_stage,
            recall_at_fetch_k=recall_fetch,
            recall_at_top_k=recall_top,
        )
        if TRACE_HYDE_DOCUMENT_KEY in meta:
            trace["hyde_document"] = meta[TRACE_HYDE_DOCUMENT_KEY]
        if reranked_chunks is not None:
            trace["reranked"] = _to_stage_chunks(reranked_chunks, case)

        traces.append(trace)
        per_case_summary.append(
            SearchCaseResult(
                query=case["query"],
                recall_at_k=recall_top,
                top_preview=[c.content[:120] for c in final_chunks[:resolved_top_k]],
            )
        )

    await indexer.store.aclose()

    summary = SearchDatasetSummary(
        dataset=dataset,
        profile=profile.profile_id,
        collection=coll,
        top_k=resolved_top_k,
        fetch_k=fetch_k,
        cases=len(cases),
        mean_recall_at_top_k=mean_recall_at_k(top_k_scores),
        mean_recall_at_fetch_k=mean_recall_at_k(fetch_k_scores),
        timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
        per_case=per_case_summary,
    )
    return summary, traces


async def search_dataset(
    *,
    dataset: str,
    profile: RAGProfile,
    in_memory: bool,
    top_k: int | None = None,
    recall_n: int | None = None,
    collection: str | None = None,
) -> SearchDatasetSummary:
    """Lightweight search eval without per-stage traces (backward-compatible)."""
    retriever_cfg = get_rag_config().retriever
    resolved_top_k = top_k if top_k is not None else retriever_cfg.top_k
    resolved_recall_n = recall_n if recall_n is not None else retriever_cfg.recall_n
    coll = collection or collection_name(dataset, profile.profile_id)
    from eval.profiles import build_indexer_for_profile

    indexer = build_indexer_for_profile(profile, coll, in_memory=in_memory)
    retriever = build_retriever_for_profile(
        profile,
        coll,
        in_memory=in_memory,
        store=indexer.store,
        embedder=indexer.embedder,
        recall_n=resolved_recall_n,
    )

    gold_path = dataset_dir(dataset) / "gold_rag.jsonl"
    cases = load_gold_cases(gold_path)
    scores: list[float] = []
    per_case: list[SearchCaseResult] = []

    for case in cases:
        chunks = await retriever.aquery(case["query"], top_k=resolved_top_k)
        score = recall_at_k(chunks, case, k=resolved_top_k)
        scores.append(score)
        per_case.append(
            SearchCaseResult(
                query=case["query"],
                recall_at_k=score,
                top_preview=[c.content[:120] for c in chunks[:resolved_top_k]],
            )
        )

    await indexer.store.aclose()

    # fetch_k not applicable here; fill with top_k to keep TypedDict valid
    return SearchDatasetSummary(
        dataset=dataset,
        profile=profile.profile_id,
        collection=coll,
        top_k=resolved_top_k,
        fetch_k=resolved_top_k,
        cases=len(cases),
        mean_recall_at_top_k=mean_recall_at_k(scores),
        mean_recall_at_fetch_k=mean_recall_at_k(scores),
        timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
        per_case=per_case,
    )


def _write_trace_results(
    out_dir: Path,
    summary: SearchDatasetSummary,
    traces: list[QueryTrace],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    traces_path = out_dir / "traces.jsonl"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with traces_path.open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"summary → {summary_path}")
    print(f"traces  → {traces_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Search eval gold queries against Qdrant.")
    parser.add_argument("--dataset", default="smoke")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--collection", default=None, help="Override Qdrant collection name")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--recall-n", type=int, default=None)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Collect per-stage traces and write traces.jsonl + summary.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Directory to write trace results (default: "
            "eval/results/{dataset}_{profile}_{timestamp}). Implies --trace."
        ),
    )
    args = parser.parse_args()

    profile = RAGProfile.get(args.profile)
    run_trace = args.trace or args.out is not None

    if run_trace:
        summary, traces = asyncio.run(
            trace_dataset(
                dataset=args.dataset,
                profile=profile,
                in_memory=args.in_memory,
                top_k=args.top_k,
                recall_n=args.recall_n,
                collection=args.collection,
            )
        )
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_dir = (
            Path(args.out)
            if args.out
            else Path("eval/results") / f"{args.dataset}_{args.profile}_{ts}"
        )
        _write_trace_results(out_dir, summary, traces)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(
                f"mean_recall@top_k={summary['mean_recall_at_top_k']:.3f}  "
                f"mean_recall@fetch_k={summary['mean_recall_at_fetch_k']:.3f}  "
                f"cases={summary['cases']}"
            )
    else:
        result = asyncio.run(
            search_dataset(
                dataset=args.dataset,
                profile=profile,
                in_memory=args.in_memory,
                top_k=args.top_k,
                recall_n=args.recall_n,
                collection=args.collection,
            )
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result)


if __name__ == "__main__":
    main()
