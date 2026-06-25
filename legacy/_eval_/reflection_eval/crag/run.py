"""CLI entry point: CRAG reflection eval on a pooled BEIR dataset.

Examples::

    python -m _eval_.reflection_eval.crag.run --dataset trec-covid --limit 5
    python -m _eval_.reflection_eval.crag.run --dataset trec-covid --limit 50 --judge
    python -m _eval_.reflection_eval.crag.run --dataset trec-covid --limit 50 --no-recreate
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from _eval_.paths import REPO_ROOT
from _eval_.reflection_eval.beir_runner import (
    add_beir_cli_args,
    agent_run_config_from_args,
    write_reflection_results,
)
from _eval_.reflection_eval.crag.pipeline import CRAG_PATTERN, RESULT_PREFIX, evaluate_crag, format_crag_table

load_dotenv(REPO_ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRAG reflection eval (react vs react_crag).")
    add_beir_cli_args(parser)
    args = parser.parse_args()

    cfg = agent_run_config_from_args(args, pattern=CRAG_PATTERN)

    async def _run() -> None:
        result, indexer = await evaluate_crag(cfg)
        try:
            print("\n" + format_crag_table(result))
            out_path = write_reflection_results(
                prefix=RESULT_PREFIX,
                dataset=cfg.dataset,
                config_payload={
                    "pattern": CRAG_PATTERN,
                    "baseline": result.baseline.pattern,
                    "rag_profile": cfg.rag_profile,
                    "k_values": list(cfg.k_values),
                    "query_limit": cfg.query_limit,
                    "agent_rag_top_k": cfg.agent_rag_top_k,
                    "rel_threshold": cfg.pool_spec.rel_threshold,
                    "max_distractors_per_query": cfg.pool_spec.max_distractors_per_query,
                    "use_judge": cfg.use_judge,
                    "recreate": cfg.recreate,
                },
                baseline=result.baseline,
                pattern=result.pattern,
                signal_metrics=result.signal_metrics,
                deltas=result.deltas,
            )
            print(f"\nresults -> {out_path}")
        finally:
            await indexer.store.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
