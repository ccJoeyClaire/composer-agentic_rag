"""
Smoke eval: index + search for all RAG profiles on a dataset.

Usage (repo root):
  python -m eval.run_smoke --dataset smoke
  python -m eval.run_smoke --dataset smoke --in-memory
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.paths import REPO_ROOT as ROOT

from eval.profiles import SMOKE_PROFILES
from eval.run_index import index_dataset
from eval.run_search import search_dataset


def _format_results_table(rows: list[dict]) -> str:
    lines = [
        "| Profile | Recall@3 | Chunks | Index OK |",
        "|---------|----------|--------|----------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['mean_recall_at_k']:.2f} | "
            f"{row['total_chunks']} | {'yes' if row['index_ok'] else 'no'} |"
        )
    return "\n".join(lines)


def _append_eval_results(dataset: str, rows: list[dict]) -> None:
    results_path = ROOT / "docs" / "eval_results.md"
    today = date.today().isoformat()
    block = [
        f"\n## Smoke ({dataset}) — {today}\n",
        _format_results_table(rows),
        "",
    ]
    text = results_path.read_text(encoding="utf-8") if results_path.is_file() else ""
    marker = f"## Smoke ({dataset}) — {today}"
    if marker in text:
        # Replace prior block for the same dataset/day (e.g. after fixing a failed run).
        start = text.index(marker)
        rest = text[start + len(marker) :]
        next_hdr = rest.find("\n## ")
        end = start + len(marker) + (next_hdr if next_hdr != -1 else len(rest))
        text = text[:start] + text[end:].lstrip("\n")
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))


async def run_smoke(
    *,
    dataset: str,
    in_memory: bool,
    recreate: bool,
    profiles: list[str] | None,
) -> list[dict]:
    selected = SMOKE_PROFILES
    if profiles:
        profile_set = set(profiles)
        selected = [p for p in SMOKE_PROFILES if p.profile_id in profile_set]

    rows: list[dict] = []
    for profile in selected:
        index_summary = await index_dataset(
            dataset=dataset,
            profile=profile,
            in_memory=in_memory,
            recreate=recreate,
        )
        search_summary = await search_dataset(
            dataset=dataset,
            profile=profile,
            in_memory=in_memory,
        )
        rows.append(
            {
                "profile": profile.profile_id,
                "mean_recall_at_k": search_summary["mean_recall_at_k"],
                "total_chunks": index_summary["total_chunks"],
                "index_ok": index_summary["ok"],
                "collection": index_summary["collection"],
            }
        )
        print(
            f"[{profile.profile_id}] index_ok={index_summary['ok']} "
            f"chunks={index_summary['total_chunks']} "
            f"recall@3={search_summary['mean_recall_at_k']:.2f}"
        )

    print("\n" + _format_results_table(rows))
    _append_eval_results(dataset, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke RAG eval for all profiles.")
    parser.add_argument("--dataset", default="smoke")
    parser.add_argument("--profile", action="append", dest="profiles", help="Subset of profiles")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--recreate", action="store_true", help="Drop collection before each index")
    args = parser.parse_args()

    rows = asyncio.run(
        run_smoke(
            dataset=args.dataset,
            in_memory=args.in_memory,
            recreate=args.recreate,
            profiles=args.profiles,
        )
    )
    if not rows or any(not r["index_ok"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
