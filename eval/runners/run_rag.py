"""
RAG profile smoke: index + search all profiles on the smoke dataset.

Usage:
  python -m eval.runners.run_rag --recreate
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.paths import REPO_ROOT as ROOT, SMOKE_DATASET_NAME
from eval.profiles import SMOKE_PROFILES
from eval.rag.run_index import index_dataset
from eval.rag.run_search import search_dataset


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


def _append_eval_results(rows: list[dict]) -> None:
    results_path = ROOT / "docs" / "eval_results.md"
    today = date.today().isoformat()
    block = [
        f"\n## RAG Smoke — {today}\n",
        _format_results_table(rows),
        "",
    ]
    text = results_path.read_text(encoding="utf-8") if results_path.is_file() else ""
    marker = f"## RAG Smoke — {today}"
    if marker in text:
        start = text.index(marker)
        rest = text[start + len(marker) :]
        next_hdr = rest.find("\n## ")
        end = start + len(marker) + (next_hdr if next_hdr != -1 else len(rest))
        text = text[:start] + text[end:].lstrip("\n")
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


async def run_rag_smoke(
    *,
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
            dataset=SMOKE_DATASET_NAME,
            profile=profile,
            in_memory=in_memory,
            recreate=recreate,
        )
        search_summary = await search_dataset(
            dataset=SMOKE_DATASET_NAME,
            profile=profile,
            in_memory=in_memory,
        )
        rows.append(
            {
                "profile": profile.profile_id,
                "mean_recall_at_k": search_summary["mean_recall_at_top_k"],
                "total_chunks": index_summary["total_chunks"],
                "index_ok": index_summary["ok"],
                "collection": index_summary["collection"],
            }
        )
        print(
            f"[{profile.profile_id}] index_ok={index_summary['ok']} "
            f"chunks={index_summary['total_chunks']} "
            f"recall@3={search_summary['mean_recall_at_top_k']:.2f}"
        )

    print("\n" + _format_results_table(rows))
    _append_eval_results(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG smoke eval for all profiles.")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    rows = asyncio.run(
        run_rag_smoke(
            in_memory=args.in_memory,
            recreate=args.recreate,
            profiles=args.profiles,
        )
    )
    if not rows or any(not r["index_ok"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
