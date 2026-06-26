"""Replay the query pipeline against an existing indexed collection.

Library usage::

    from _eval_.analysis.replay.query import QueryReplayRequest, replay_query

    result = await replay_query(
        QueryReplayRequest(
            profile="semantic_rerank",
            dataset="nfcorpus",
            query_id="PLAIN-2",
            top_k=80,
        )
    )

Run (offline fixture demo, no Qdrant / LLM)::

    python -m _eval_.analysis.replay.query

CLI (live replay, pass any flag)::

    python -m _eval_.analysis.replay.query \\
      --profile semantic_rerank \\
      --dataset nfcorpus \\
      --query-id PLAIN-2 \\
      --top-k 80 \\
      --stages retrieved,reranked,final
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TypedDict

from rag.base import Chunk, TRACE_RERANKED_KEY, TRACE_RETRIEVED_KEY

from _eval_.config import DATASETS
from _eval_.data_preparing.beir import DocId, load_qrels, load_queries
from _eval_.data_preparing.pooling import gold_docs
from _eval_.reflection_eval.beir_runner import load_profile_flags
from _eval_.analysis.replay.rank_report import (
    GoldRankRow,
    StageName,
    format_gold_table,
    format_stage_table,
    gold_rank_report,
)
from _eval_.analysis.replay.rag_factory import open_retriever, resolve_collection

_DEFAULT_REL_THRESHOLD = 1
_DEFAULT_STAGES: tuple[StageName, ...] = ("retrieved", "reranked", "final")


class StageChunkPreview(TypedDict):
    doc_id: str | None
    chunk_role: str | None
    score: float
    content_preview: str


class QueryReplayResult(TypedDict):
    """Structured outcome of :func:`replay_query`."""

    profile: str
    collection: str
    query: str
    top_k: int
    has_reranker: bool
    retrieved: list[Chunk]
    reranked: list[Chunk] | None
    final: list[Chunk]
    gold_report: list[GoldRankRow]


@dataclass(frozen=True)
class QueryReplayRequest:
    """Inputs for :func:`replay_query` — explicit, importable, no argparse."""

    profile: str
    dataset: str = "nfcorpus"
    collection: str | None = None
    query: str | None = None
    query_id: str | None = None
    top_k: int = 80
    stages: tuple[StageName, ...] = _DEFAULT_STAGES
    gold_doc_ids: frozenset[DocId] = field(default_factory=frozenset)
    hyde_from_log: str | None = None


def parse_stages(raw: str) -> tuple[StageName, ...]:
    """Parse comma-separated stage names for replay output."""
    stages: list[StageName] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if name not in {"retrieved", "reranked", "final"}:
            raise ValueError(f"unknown stage {name!r}; use retrieved,reranked,final")
        stages.append(name)  # type: ignore[arg-type]
    return tuple(stages)


def resolve_query_text(
    *,
    dataset: str,
    query: str | None,
    query_id: str | None,
) -> str:
    """Resolve query string from inline text or a BEIR query id."""
    if query:
        return query.strip()
    if not query_id:
        raise ValueError("Provide query text or query_id.")
    queries = load_queries(DATASETS[dataset].queries_path())
    if query_id not in queries:
        raise ValueError(f"Unknown query id {query_id!r} for dataset {dataset!r}.")
    return queries[query_id].text


def resolve_gold_doc_ids(
    *,
    dataset: str,
    query_id: str | None,
    gold_doc_ids: frozenset[DocId],
    rel_threshold: int = _DEFAULT_REL_THRESHOLD,
) -> set[DocId]:
    """Resolve gold doc ids from explicit ids or qrels for ``query_id``."""
    if gold_doc_ids:
        return set(gold_doc_ids)
    if not query_id:
        return set()
    qrels = load_qrels(DATASETS[dataset].qrels_path())
    if query_id not in qrels:
        raise ValueError(f"No qrels for query id {query_id!r}.")
    return gold_docs(qrels[query_id], rel_threshold)


def chunk_to_stage_preview(chunk: Chunk) -> StageChunkPreview:
    meta = chunk.metadata or {}
    return StageChunkPreview(
        doc_id=meta.get("doc_id"),
        chunk_role=meta.get("chunk_role"),
        score=chunk.score,
        content_preview=chunk.content[:80],
    )


async def replay_query(request: QueryReplayRequest) -> QueryReplayResult:
    """Run the profile retriever against an existing collection and capture trace stages.

    Args:
        request: Profile, collection resolution, query text/id, and replay knobs.

    Returns:
        Per-stage chunk lists plus optional gold-doc rank report.
    """
    if request.hyde_from_log:
        print(
            "Note: hyde_from_log is not implemented yet (phase 2); using live HyDE if enabled.",
            file=sys.stderr,
        )

    collection = resolve_collection(request.dataset, request.profile, request.collection)
    query_text = resolve_query_text(
        dataset=request.dataset,
        query=request.query,
        query_id=request.query_id,
    )
    gold_ids = resolve_gold_doc_ids(
        dataset=request.dataset,
        query_id=request.query_id,
        gold_doc_ids=request.gold_doc_ids,
    )

    retriever, store = open_retriever(request.profile, collection)
    try:
        trace = await retriever.aquery_trace(query_text, top_k=request.top_k)
    finally:
        await store.aclose()

    retrieved = list(trace.metadata.get(TRACE_RETRIEVED_KEY) or [])
    reranked_raw = trace.metadata.get(TRACE_RERANKED_KEY)
    reranked = list(reranked_raw) if reranked_raw is not None else None
    final = list(trace.chunks)
    has_reranker = load_profile_flags(request.profile)["use_reranker"]

    gold_report = (
        gold_rank_report(
            gold_ids,
            retrieved=retrieved,
            reranked=reranked if has_reranker else None,
            final=final,
            top_k=request.top_k,
        )
        if gold_ids
        else []
    )

    return QueryReplayResult(
        profile=request.profile,
        collection=collection,
        query=query_text,
        top_k=request.top_k,
        has_reranker=has_reranker,
        retrieved=retrieved,
        reranked=reranked,
        final=final,
        gold_report=gold_report,
    )


def format_query_replay_result(
    result: QueryReplayResult,
    *,
    stages: tuple[StageName, ...] = _DEFAULT_STAGES,
) -> str:
    """Human-readable multi-stage table for CLI or notebooks."""
    lines = [
        f"profile={result['profile']} collection={result['collection']}",
        f"query: {result['query']}",
        "",
    ]
    retrieved = result["retrieved"]
    reranked = result["reranked"]
    final = result["final"]
    has_reranker = result["has_reranker"]

    for stage in stages:
        if stage == "reranked" and not has_reranker:
            lines.append("[reranked] (skipped — profile has use_reranker=false)")
            lines.append("")
            continue
        if stage == "reranked" and reranked is None:
            lines.append("[reranked] (no reranker stage in trace)")
            lines.append("")
            continue
        chunks = (
            retrieved
            if stage == "retrieved"
            else (reranked if stage == "reranked" else final)
        )
        lines.append(format_stage_table(chunks, stage=stage))
        lines.append("")

    if result["gold_report"]:
        lines.append(format_gold_table(result["gold_report"]))

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import asyncio
    import json

    def _query_replay_result_to_json_dict(
        result: QueryReplayResult,
        *,
        stages: tuple[StageName, ...] = _DEFAULT_STAGES,
    ) -> dict[str, object]:
        stage_chunks = {
            "retrieved": result["retrieved"],
            "reranked": result["reranked"] or [],
            "final": result["final"],
        }
        has_reranker = result["has_reranker"]
        payload: dict[str, object] = {
            "profile": result["profile"],
            "collection": result["collection"],
            "query": result["query"],
            "top_k": result["top_k"],
            "stages": {
                stage: [chunk_to_stage_preview(chunk) for chunk in stage_chunks[stage]]
                for stage in stages
                if stage != "reranked" or has_reranker
            },
        }
        if result["gold_report"]:
            payload["gold_report"] = result["gold_report"]
        return payload

    def _request_from_cli(args: argparse.Namespace) -> QueryReplayRequest:
        gold_ids: frozenset[DocId] = frozenset()
        if args.gold_doc_ids:
            gold_ids = frozenset(
                doc_id.strip() for doc_id in args.gold_doc_ids.split(",") if doc_id.strip()
            )
        return QueryReplayRequest(
            profile=args.profile,
            dataset=args.dataset,
            collection=args.collection,
            query=args.query,
            query_id=args.query_id,
            top_k=args.top_k,
            stages=parse_stages(args.stages),
            gold_doc_ids=gold_ids,
            hyde_from_log=args.hyde_from_log,
        )

    async def _run_cli(
        request: QueryReplayRequest,
        *,
        as_json: bool = False,
    ) -> int:
        result = await replay_query(request)
        if as_json:
            print(
                json.dumps(
                    _query_replay_result_to_json_dict(result, stages=request.stages),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(format_query_replay_result(result, stages=request.stages))
        return 0

    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Replay query pipeline against an existing eval collection.",
        )
        parser.add_argument("--profile", required=True, help="arg_config.yaml profile id")
        parser.add_argument("--collection", help="Override Qdrant collection name")
        parser.add_argument("--dataset", default="nfcorpus", choices=sorted(DATASETS))
        parser.add_argument("--query", help="Query text")
        parser.add_argument("--query-id", help="BEIR query id (loads text + optional qrels gold)")
        parser.add_argument(
            "--top-k", type=int, default=80, help="Final top-k passed to aquery_trace"
        )
        parser.add_argument(
            "--stages",
            default="retrieved,reranked,final",
            help="Comma-separated stages to print",
        )
        parser.add_argument(
            "--gold-doc-ids",
            help="Comma-separated gold doc ids (overrides qrels lookup)",
        )
        parser.add_argument(
            "--hyde-from-log",
            dest="hyde_from_log",
            help="Phase 2: replay HyDE text from JSONL instead of calling the LLM",
        )
        parser.add_argument("--json", action="store_true", help="Emit JSON instead of tables")
        return parser

    def main(argv: list[str] | None = None) -> int:
        args = _build_parser().parse_args(argv)
        try:
            request = _request_from_cli(args)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        return asyncio.run(_run_cli(request, as_json=args.json))

    if len(sys.argv) > 1:
        raise SystemExit(main())

    def _demo_trace_chunks() -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
        """Synthetic aquery_trace stages for offline demo (no Qdrant / LLM)."""
        retrieved = [
            Chunk(
                "noise passage about unrelated topic",
                metadata={"doc_id": "distractor", "chunk_role": "small"},
                score=0.91,
            ),
            Chunk(
                "relevant passage about vitamin D",
                metadata={"doc_id": "gold_doc", "chunk_role": "small"},
                score=0.88,
            ),
            Chunk(
                "another distractor",
                metadata={"doc_id": "other", "chunk_role": "small"},
                score=0.85,
            ),
        ]
        reranked = [
            Chunk(
                "relevant passage about vitamin D",
                metadata={"doc_id": "gold_doc", "chunk_role": "small"},
                score=0.97,
            ),
            Chunk(
                "noise passage about unrelated topic",
                metadata={"doc_id": "distractor", "chunk_role": "small"},
                score=0.42,
            ),
            Chunk(
                "another distractor",
                metadata={"doc_id": "other", "chunk_role": "small"},
                score=0.31,
            ),
        ]
        final = reranked[:2]
        return retrieved, reranked, final

    def _demo_query_replay_result() -> QueryReplayResult:
        retrieved, reranked, final = _demo_trace_chunks()
        return QueryReplayResult(
            profile="semantic_rerank",
            collection="demo_offline",
            query="Do vitamin D supplements help immunity?",
            top_k=2,
            has_reranker=True,
            retrieved=retrieved,
            reranked=reranked,
            final=final,
            gold_report=gold_rank_report(
                {"gold_doc", "missing_gold"},
                retrieved=retrieved,
                reranked=reranked,
                final=final,
                top_k=2,
            ),
        )

    live_request = QueryReplayRequest(
        profile="semantic_rerank",
        dataset="nfcorpus",
        query_id="PLAIN-2",
        top_k=80,
        stages=("retrieved", "reranked", "final"),
    )
    print("=== QueryReplayRequest (pass to replay_query) ===")
    print(f"  profile={live_request.profile!r}")
    print(f"  dataset={live_request.dataset!r}")
    print(f"  query_id={live_request.query_id!r}")
    print(f"  top_k={live_request.top_k}")
    print(f"  stages={live_request.stages}")

    result = _demo_query_replay_result()
    print("\n=== format_query_replay_result (synthetic trace) ===\n")
    print(format_query_replay_result(result, stages=live_request.stages))

    print(
        "\nLive replay (indexed collection + API keys):\n"
        "  python -m _eval_.analysis.replay.query "
        "--profile semantic_rerank --query-id PLAIN-2"
    )
