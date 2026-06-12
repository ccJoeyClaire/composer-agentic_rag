"""
Agent smoke eval: mock behavior cases per pattern (no real LLM).

Usage (repo root):
  python -m eval.run_agent --dataset smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.metrics.agent_checks import run_all_cases
from eval.metrics.recall import load_gold_cases
from eval.paths import dataset_dir


def _load_agent_cases(dataset: str) -> list[dict]:
    path = dataset_dir(dataset) / "gold_agent.jsonl"
    return load_gold_cases(path)


async def run_agent_eval(dataset: str) -> dict:
    cases = _load_agent_cases(dataset)
    results = await run_all_cases(cases)
    passed = sum(1 for r in results if r.get("ok"))
    return {
        "dataset": dataset,
        "cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent mock eval cases.")
    parser.add_argument("--dataset", default="smoke")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = asyncio.run(run_agent_eval(args.dataset))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for row in summary["results"]:
            status = "PASS" if row.get("ok") else "FAIL"
            print(f"[{status}] {row.get('case_id')}: {row}")
        print(f"\nPassed {summary['passed']}/{summary['cases']}")

    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
