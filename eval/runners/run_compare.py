"""
Smoke: index baseline + compare agent context vs baseline RAG.

Usage:
  python -m eval.runners.run_compare --recreate
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from eval.agent.run_context import _format_table, run_agent_context_eval
from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.paths import REPO_ROOT as ROOT, SMOKE_DATASET_NAME
from eval.profiles import RAGProfile
from eval.rag.run_index import index_dataset


def _append_eval_results(summary: dict) -> None:
    results_path = ROOT / "docs" / "eval_results.md"
    today = date.today().isoformat()
    block = [
        f"\n## Agent Context vs Baseline — {today}\n",
        f"profile={summary['rag_profile']} collection={summary['collection']}\n",
        _format_table(summary),
        "",
    ]
    text = results_path.read_text(encoding="utf-8") if results_path.is_file() else ""
    marker = f"## Agent Context vs Baseline — {today}"
    if marker in text:
        start = text.index(marker)
        rest = text[start + len(marker) :]
        next_hdr = rest.find("\n## ")
        end = start + len(marker) + (next_hdr if next_hdr != -1 else len(rest))
        text = text[:start] + text[end:].lstrip("\n")
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


async def run_compare_smoke(*, in_memory: bool, recreate: bool) -> dict:
    profile = RAGProfile.get("baseline")
    index_summary = await index_dataset(
        dataset=SMOKE_DATASET_NAME,
        profile=profile,
        in_memory=in_memory,
        recreate=recreate,
    )
    if not index_summary["ok"]:
        raise RuntimeError(f"Index failed: {index_summary}")

    summary = await run_agent_context_eval(
        dataset=SMOKE_DATASET_NAME,
        profile=profile,
        in_memory=in_memory,
    )
    print("\n" + _format_table(summary))
    _append_eval_results(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Index + agent-vs-baseline context compare.")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run_compare_smoke(in_memory=args.in_memory, recreate=args.recreate)
    )


if __name__ == "__main__":
    main()
