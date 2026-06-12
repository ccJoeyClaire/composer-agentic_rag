"""
Run gold RAG queries against an indexed collection.

Usage (repo root):
  python -m eval.run_search --dataset smoke --profile baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.metrics.recall import load_gold_cases, mean_recall_at_k, recall_at_k
from eval.paths import dataset_dir
from eval.profiles import RAGProfile, build_retriever_for_profile, collection_name


async def search_dataset(
    *,
    dataset: str,
    profile: RAGProfile,
    in_memory: bool,
    top_k: int = 3,
    recall_n: int = 50,
    collection: str | None = None,
) -> dict:
    coll = collection or collection_name(dataset, profile.profile_id)
    from eval.profiles import build_indexer_for_profile

    indexer = build_indexer_for_profile(profile, coll, in_memory=in_memory)
    retriever = build_retriever_for_profile(
        profile,
        coll,
        in_memory=in_memory,
        store=indexer.store,
        embedder=indexer.embedder,
        recall_n=recall_n,
    )

    gold_path = dataset_dir(dataset) / "gold_rag.jsonl"
    cases = load_gold_cases(gold_path)
    scores: list[float] = []
    per_case: list[dict] = []

    for case in cases:
        chunks = await retriever.aquery(case["query"], top_k=top_k)
        score = recall_at_k(chunks, case, k=top_k)
        scores.append(score)
        per_case.append(
            {
                "query": case["query"],
                "recall_at_k": score,
                "top_preview": [c.content[:120] for c in chunks[:top_k]],
            }
        )

    await indexer.store.aclose()

    mean = mean_recall_at_k(scores)
    return {
        "dataset": dataset,
        "profile": profile.profile_id,
        "collection": coll,
        "top_k": top_k,
        "cases": len(cases),
        "mean_recall_at_k": mean,
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search eval gold queries against Qdrant.")
    parser.add_argument("--dataset", default="smoke")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--collection", default=None, help="Override Qdrant collection name")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recall-n", type=int, default=50)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    profile = RAGProfile.get(args.profile)
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
