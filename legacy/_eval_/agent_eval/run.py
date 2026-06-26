"""CLI entry point: evaluate agent reflection patterns on a pooled BEIR dataset.

All patterns share one pooled index (served by a single RAG profile), so the
table isolates the effect of agent control flow (react vs crag vs self_rag ...)
on context recall, process cost, and (optionally) judged answer quality.

Examples (run from repo root):

    # Compare plain ReAct vs CRAG vs Self-RAG on 5 queries.
    python -m _eval_.agent_eval.run --dataset trec-covid \
        --pattern react --pattern react_crag --pattern react_self_rag

    # Add LLM-judged answer correctness/grounding.
    python -m _eval_.agent_eval.run --dataset trec-covid --pattern react --judge
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from _eval_.paths import REPO_ROOT, results_dir

load_dotenv(REPO_ROOT / ".env")

import asyncio

import yaml

from rag.build import build_RAG_indexer, build_RAG_retriever
from rag.core import RAGIndexer, RAGRetriever

from _eval_.agent_eval.pipeline import (
    AgentPatternResult,
    drop_collection,
    evaluate_pattern,
    index_doc_list,
)
from _eval_.config import (
    DATASETS,
    DEFAULT_AGENT_RAG_TOP_K,
    DEFAULT_PREDICT_QUESTION_MAX_CONCURRENCY,
    DEFAULT_QUERY_LIMIT,
    AgentRunConfig,
    collection_name,
)
from _eval_.data_preparing.beir import QueryId, iter_corpus, load_qrels, load_queries
from _eval_.data_preparing.pooling import PoolSpec, build_pool, gold_docs, queries_with_gold

_MAX_GOLD_TEXTS_PER_QUERY = 3
_PROFILE_BOOL_FIELDS = (
    "use_token_chunker",
    "use_contextual",
    "use_small_to_big",
    "use_predict_questions",
    "use_hyde",
    "use_reranker",
)


def _load_profile_flags(profile_id: str) -> dict[str, bool]:
    with (REPO_ROOT / "arg_config.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    raw = data["profiles"][profile_id]
    return {key: bool(raw.get(key, False)) for key in _PROFILE_BOOL_FIELDS}


def _build_agent_rag(profile_id: str, collection: str) -> tuple[RAGIndexer, RAGRetriever]:
    """Build indexer + retriever for the agent's shared RAG backend."""
    flags = _load_profile_flags(profile_id)
    predict_concurrency = (
        max(1, int(os.environ.get("EVAL_LLM_CONCURRENCY", DEFAULT_PREDICT_QUESTION_MAX_CONCURRENCY)))
        if flags["use_predict_questions"]
        else None
    )
    indexer = build_RAG_indexer(
        collection,
        use_token_chunker=flags["use_token_chunker"],
        use_contextual=flags["use_contextual"],
        use_predict_questions=flags["use_predict_questions"],
        use_small_to_big=flags["use_small_to_big"],
        predict_question_max_concurrency=predict_concurrency,
    )
    retriever = build_RAG_retriever(
        collection,
        use_reranker=flags["use_reranker"],
        use_contextual=flags["use_contextual"],
        use_hyde=flags["use_hyde"],
        use_small_to_big=flags["use_small_to_big"],
        store=indexer.store,
        embedder=indexer.embedder,
    )
    return indexer, retriever


def _format_table(results: list[AgentPatternResult]) -> str:
    if not results:
        return "(no results)"
    metric_keys = list(results[0].mean_metrics.keys())
    header = ["pattern", "queries", *metric_keys]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for res in results:
        cells = [
            res.pattern,
            str(res.num_queries),
            *(f"{res.mean_metrics[k]:.3f}" for k in metric_keys),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_results(cfg: AgentRunConfig, results: list[AgentPatternResult]) -> str:
    timestamp = datetime.now(tz=timezone.utc)
    out_path = results_dir() / f"agent_{cfg.dataset}_{timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    payload = {
        "dataset": cfg.dataset,
        "timestamp_utc": timestamp.isoformat(),
        "config": {
            "patterns": cfg.patterns,
            "rag_profile": cfg.rag_profile,
            "k_values": list(cfg.k_values),
            "query_limit": cfg.query_limit,
            "agent_rag_top_k": cfg.agent_rag_top_k,
            "rel_threshold": cfg.pool_spec.rel_threshold,
            "max_distractors_per_query": cfg.pool_spec.max_distractors_per_query,
            "use_judge": cfg.use_judge,
        },
        "results": [dataclasses.asdict(res) for res in results],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


async def run_agent_eval(cfg: AgentRunConfig) -> list[AgentPatternResult]:
    spec = DATASETS[cfg.dataset]
    qrels = load_qrels(spec.qrels_path())
    queries = load_queries(spec.queries_path())

    candidate_ids = queries_with_gold(qrels, cfg.pool_spec.rel_threshold)
    query_ids = [qid for qid in candidate_ids if qid in queries]
    if cfg.query_limit is not None:
        query_ids = query_ids[: cfg.query_limit]
    if not query_ids:
        raise SystemExit("No evaluable queries (no gold under threshold / id mismatch).")

    pool_ids = build_pool(qrels, query_ids, cfg.pool_spec)
    docs = list(iter_corpus(spec.corpus_path(), keep_ids=pool_ids))
    text_by_id = {doc.doc_id: doc.text for doc in docs}
    gold_texts = _gold_texts(query_ids, qrels, text_by_id, cfg)

    print(
        f"dataset={cfg.dataset} queries={len(query_ids)} pool_docs={len(docs)} "
        f"rag_profile={cfg.rag_profile} patterns={cfg.patterns} judge={cfg.use_judge}"
    )

    collection = collection_name(cfg.dataset, cfg.rag_profile)
    indexer, retriever = _build_agent_rag(cfg.rag_profile, collection)
    if cfg.recreate:
        await drop_collection(indexer.store)
    await index_doc_list(indexer, docs, concurrency=cfg.index_concurrency)

    results: list[AgentPatternResult] = []
    try:
        for pattern in cfg.patterns:
            print(f"[{pattern}] running {len(query_ids)} queries ...")
            result = await evaluate_pattern(
                pattern,
                retriever=retriever,
                queries=queries,
                qrels=qrels,
                query_ids=query_ids,
                gold_texts=gold_texts,
                cfg=cfg,
            )
            results.append(result)
    finally:
        await indexer.store.aclose()
    return results


def _gold_texts(
    query_ids: list[QueryId],
    qrels,
    text_by_id: dict[str, str],
    cfg: AgentRunConfig,
) -> dict[QueryId, list[str]]:
    """Collect a few gold passage texts per query for the optional judge."""
    if not cfg.use_judge:
        return {}
    result: dict[QueryId, list[str]] = {}
    for qid in query_ids:
        gold = gold_docs(qrels[qid], cfg.pool_spec.rel_threshold)
        texts = [text_by_id[d] for d in gold if d in text_by_id]
        result[qid] = texts[:_MAX_GOLD_TEXTS_PER_QUERY]
    return result


def _build_config(args: argparse.Namespace) -> AgentRunConfig:
    patterns = args.patterns or ["react"]
    k_values = tuple(sorted({int(k) for k in args.k})) if args.k else (3, 10, 20)
    pool_spec = PoolSpec(
        rel_threshold=args.rel_threshold,
        max_distractors_per_query=args.max_distractors,
    )
    return AgentRunConfig(
        dataset=args.dataset,
        patterns=patterns,
        rag_profile=args.rag_profile,
        pool_spec=pool_spec,
        k_values=k_values,
        query_limit=args.limit,
        agent_rag_top_k=args.agent_rag_top_k,
        recreate=not args.no_recreate,
        use_judge=args.judge,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pooled BEIR agent eval (context + process + judge).")
    parser.add_argument("--dataset", default="trec-covid")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Repeatable; default: react")
    parser.add_argument("--rag-profile", default="baseline", help="RAG profile backing the agent's tool")
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Max queries (None=all)")
    parser.add_argument("--rel-threshold", type=int, default=1, help="Min judged score counted as gold")
    parser.add_argument("--max-distractors", type=int, default=None, help="Cap judged-irrelevant docs per query")
    parser.add_argument("--k", action="append", help="Repeatable metric cutoff (default 3,10,20)")
    parser.add_argument("--agent-rag-top-k", type=int, default=DEFAULT_AGENT_RAG_TOP_K)
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge answer scoring")
    parser.add_argument("--no-recreate", action="store_true", help="Keep existing collection (faster re-runs)")
    args = parser.parse_args()

    cfg = _build_config(args)
    results = asyncio.run(run_agent_eval(cfg))

    print("\n" + _format_table(results))
    out_path = _write_results(cfg, results)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
