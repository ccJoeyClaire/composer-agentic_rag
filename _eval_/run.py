"""CLI entry point: orchestrate a pooled BEIR eval and print a per-profile table.

Examples (run from repo root):

    # Cheap smoke run: 5 queries, baseline profile, local Qdrant (no Docker).
    python -m _eval_.run --dataset trec-covid

    # Compare profiles, more queries, strict relevance, cap index size.
    python -m _eval_.run --dataset trec-covid \
        --profile baseline --profile reranker --profile full \
        --limit 20 --rel-threshold 2 --max-distractors 100
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone

from eval.bootstrap import setup_eval_env

setup_eval_env()

import asyncio

from eval.profiles import RAGProfile

from _eval_.beir import load_qrels, load_queries
from _eval_.config import (
    DEFAULT_INDEX_CONCURRENCY,
    DEFAULT_QUERY_LIMIT,
    RunConfig,
    get_dataset_spec,
)
from _eval_.paths import results_dir
from _eval_.pipeline import ProfileResult, evaluate_profile
from _eval_.pooling import PoolSpec, build_pool, queries_with_gold


def _format_table(results: list[ProfileResult]) -> str:
    """Render a markdown table: one row per profile, one column per metric."""
    if not results:
        return "(no results)"
    metric_keys = list(results[0].mean_metrics.keys())
    header = ["profile", "docs", "queries", *metric_keys]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for res in results:
        cells = [
            res.profile_id,
            str(res.num_docs_indexed),
            str(res.num_queries),
            *(f"{res.mean_metrics[k]:.3f}" for k in metric_keys),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_results(cfg: RunConfig, results: list[ProfileResult]) -> str:
    timestamp = datetime.now(tz=timezone.utc)
    out_path = results_dir() / f"{cfg.dataset}_{timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    payload = {
        "dataset": cfg.dataset,
        "timestamp_utc": timestamp.isoformat(),
        "config": {
            "profiles": cfg.profiles,
            "k_values": list(cfg.k_values),
            "query_limit": cfg.query_limit,
            "rel_threshold": cfg.pool_spec.rel_threshold,
            "max_distractors_per_query": cfg.pool_spec.max_distractors_per_query,
            "fetch_chunks": cfg.fetch_chunks,
            "in_memory": cfg.in_memory,
        },
        "results": [dataclasses.asdict(res) for res in results],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


async def run_eval(cfg: RunConfig) -> list[ProfileResult]:
    spec = get_dataset_spec(cfg.dataset)
    qrels = load_qrels(spec.qrels_path())
    queries = load_queries(spec.queries_path())

    # Only queries that have gold AND a query string can be scored.
    candidate_ids = queries_with_gold(qrels, cfg.pool_spec.rel_threshold)
    query_ids = [qid for qid in candidate_ids if qid in queries]
    if cfg.query_limit is not None:
        query_ids = query_ids[: cfg.query_limit]
    if not query_ids:
        raise SystemExit("No evaluable queries (no gold under threshold / id mismatch).")

    pool_ids = build_pool(qrels, query_ids, cfg.pool_spec)
    print(
        f"dataset={cfg.dataset} queries={len(query_ids)} "
        f"pool_docs={len(pool_ids)} profiles={cfg.profiles}"
    )

    results: list[ProfileResult] = []
    for profile_id in cfg.profiles:
        profile = RAGProfile.get(profile_id)
        print(f"[{profile_id}] indexing {len(pool_ids)} docs + scoring {len(query_ids)} queries ...")
        result = await evaluate_profile(
            profile,
            corpus_path=spec.corpus_path(),
            pool_ids=pool_ids,
            queries=queries,
            qrels=qrels,
            query_ids=query_ids,
            cfg=cfg,
        )
        results.append(result)
    return results


def _build_config(args: argparse.Namespace) -> RunConfig:
    profiles = args.profiles or ["baseline"]
    k_values = tuple(sorted({int(k) for k in args.k})) if args.k else (3, 10, 20)
    pool_spec = PoolSpec(
        rel_threshold=args.rel_threshold,
        max_distractors_per_query=args.max_distractors,
    )
    return RunConfig(
        dataset=args.dataset,
        profiles=profiles,
        pool_spec=pool_spec,
        k_values=k_values,
        query_limit=args.limit,
        index_concurrency=args.index_concurrency,
        in_memory=not args.docker,
        recreate=not args.no_recreate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pooled BEIR RAG eval (Recall/MRR/nDCG).")
    parser.add_argument("--dataset", default="trec-covid", help="Dataset id (see _eval_.config.DATASETS)")
    parser.add_argument("--profile", action="append", dest="profiles", help="Repeatable; default: baseline")
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT, help="Max queries to evaluate (None=all)")
    parser.add_argument("--rel-threshold", type=int, default=1, help="Min judged score counted as gold")
    parser.add_argument("--max-distractors", type=int, default=None, help="Cap judged-irrelevant docs per query")
    parser.add_argument("--k", action="append", help="Repeatable metric cutoff (default 3,10,20)")
    parser.add_argument("--index-concurrency", type=int, default=DEFAULT_INDEX_CONCURRENCY)
    parser.add_argument("--docker", action="store_true", help="Use Qdrant at 127.0.0.1:6333 instead of local path")
    parser.add_argument("--no-recreate", action="store_true", help="Keep existing collections (faster re-runs)")
    args = parser.parse_args()

    cfg = _build_config(args)
    results = asyncio.run(run_eval(cfg))

    print("\n" + _format_table(results))
    out_path = _write_results(cfg, results)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
