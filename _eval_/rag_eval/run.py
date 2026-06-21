"""CLI entry point: orchestrate a pooled BEIR RAG eval and print a per-profile table.

Configuration is read from ``rag_eval_arg_config.yaml`` at the repo root.
Edit that file, then run::

    python -m _eval_.rag_eval.run
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from _eval_.paths import REPO_ROOT, results_dir

load_dotenv(REPO_ROOT / ".env")

from _eval_.config import RunConfig, load_rag_eval_config
from _eval_.data_preparing.prepare import prepare_eval_data
from _eval_.rag_eval.pipeline import ProfileResult, eval_pipeline


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
            "predict_question_max_concurrency": cfg.predict_question_max_concurrency,
            "chunk_fetch_multiplier": cfg.chunk_fetch_multiplier,
            "recreate": cfg.recreate,
        },
        "results": [dataclasses.asdict(res) for res in results],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


async def run_eval(cfg: RunConfig) -> list[ProfileResult]:
    data = prepare_eval_data(cfg)
    print(
        f"dataset={cfg.dataset} queries={len(data.query_ids)} "
        f"pool_docs={len(data.pool)} profiles={cfg.profiles}"
    )

    results: list[ProfileResult] = []
    for profile_id in cfg.profiles:
        index_action = (
            f"indexing {len(data.pool)} docs"
            if cfg.recreate
            else "skipping re-index when collection exists"
        )
        print(
            f"[{profile_id}] {index_action} + "
            f"scoring {len(data.query_ids)} queries ..."
        )
        result = await eval_pipeline(profile_id, data, cfg)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pooled BEIR RAG eval (index + retrieval metrics)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to rag_eval YAML (default: rag_eval_arg_config.yaml at repo root)",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Reuse existing Qdrant collection; skip re-indexing when it exists",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    cfg = load_rag_eval_config(config_path)
    if args.no_recreate:
        cfg = dataclasses.replace(cfg, recreate=False)

    results = asyncio.run(run_eval(cfg))

    print("\n" + _format_table(results))
    out_path = _write_results(cfg, results)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
