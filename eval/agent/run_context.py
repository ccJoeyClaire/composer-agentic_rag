"""
Compare baseline RAG vs agent reflection context on gold_agent_context cases.

Usage (repo root):
  python -m eval.agent.run_context --dataset smoke --profile baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.agent.compare import compare_case, summarize_compare
from eval.agent.context import load_agent_context_cases
from eval.paths import dataset_dir
from eval.profiles import RAGProfile, collection_name
from rag.config import get_rag_config


def format_table(summary: dict) -> str:
    lines = [
        "| case_id | pattern | baseline | agent | delta | raw@1 |",
        "|---------|---------|----------|-------|-------|-------|",
    ]
    for row in summary["per_case"]:
        raw_at_1 = row.get("recall_raw_fixture_at_1")
        raw_cell = f"{raw_at_1:.2f}" if raw_at_1 is not None else "-"
        lines.append(
            f"| {row['case_id']} | {row['pattern']} | "
            f"{row['recall_baseline']:.2f} | {row['recall_agent']:.2f} | "
            f"{row['delta']:+.2f} | {raw_cell} |"
        )
    lines.append(
        f"\n**mean** baseline={summary['mean_recall_baseline']:.2f} "
        f"agent={summary['mean_recall_agent']:.2f} "
        f"delta={summary['mean_delta']:+.2f}"
    )
    return "\n".join(lines)


async def run_agent_context_eval(
    *,
    dataset: str,
    profile: RAGProfile,
    in_memory: bool,
    top_k: int | None = None,
) -> dict:
    resolved_top_k = top_k if top_k is not None else get_rag_config().retriever.top_k
    coll = collection_name(dataset, profile.profile_id)
    gold_path = dataset_dir(dataset) / "gold_agent_context.jsonl"
    cases = load_agent_context_cases(gold_path)

    results = []
    for case in cases:
        row = await compare_case(
            case,
            profile,
            coll,
            in_memory=in_memory,
            top_k=resolved_top_k,
        )
        results.append(row)
        print(
            f"[{row['case_id']}] baseline={row['recall_baseline']:.2f} "
            f"agent={row['recall_agent']:.2f} delta={row['delta']:+.2f}"
        )

    summary = summarize_compare(
        dataset=dataset,
        rag_profile=profile.profile_id,
        collection=coll,
        top_k=resolved_top_k,
        results=results,
    )
    summary["timestamp_utc"] = datetime.now(tz=timezone.utc).isoformat()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline RAG recall vs agent/CRAG context recall."
    )
    parser.add_argument("--dataset", default="smoke")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    profile = RAGProfile.get(args.profile)
    summary = asyncio.run(
        run_agent_context_eval(
            dataset=args.dataset,
            profile=profile,
            in_memory=args.in_memory,
            top_k=args.top_k,
        )
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("\n" + format_table(summary))


if __name__ == "__main__":
    main()

# Backward-compatible alias used by eval.runners.run_compare
_format_table = format_table
