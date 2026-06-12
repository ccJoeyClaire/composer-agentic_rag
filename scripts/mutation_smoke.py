"""Smoke-test that key unit tests catch deliberate production-code bugs.

Run from repo root:
    python scripts/mutation_smoke.py

Each mutation temporarily patches one line in production code, runs a targeted
pytest, then restores the file. Exit code 0 only if every mutation is caught.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutation:
    path: str
    old: str
    new: str
    test: str


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "agent/subgraph/CRAG.py",
        'return "correct"',
        'return "BROKEN"',
        "tests/agent/test_crag.py::test_compute_verdict",
    ),
    Mutation(
        "rag/document_augmentation/parent_builder.py",
        "for other in matching[1:]:",
        "for other in []:  # MUTATION",
        "tests/rag/test_parent_builder.py::test_cluster_overlapping_hits_bridges_disjoint_clusters",
    ),
    Mutation(
        "rag/retriever/small_to_big_retriever.py",
        "if selected and total_tokens + est > max_total:",
        "if False:  # MUTATION",
        "tests/rag/test_small_to_big.py::test_expand_budget_limits_parent_count_over_top_k",
    ),
    Mutation(
        "agent/reflection/self_rag.py",
        "return True",
        "return False  # MUTATION",
        "tests/agent/test_self_rag.py::test_rule_based_need_retrieve",
    ),
    Mutation(
        "tests/fakes/vector_store.py",
        "return scored[:top_k]",
        "return scored  # MUTATION",
        "tests/rag/test_pipeline_integration.py::test_retriever_applies_top_k_without_reranker",
    ),
)


def _run_mutation(mutation: Mutation) -> tuple[str, str]:
    path = REPO_ROOT / mutation.path
    original = path.read_text(encoding="utf-8")
    if mutation.old not in original:
        return "SKIP", f"pattern not found in {mutation.path}"

    path.write_text(original.replace(mutation.old, mutation.new, 1), encoding="utf-8")
    result = subprocess.run(
        ["pytest", "-c", "tests/pytest.ini", mutation.test, "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    path.write_text(original, encoding="utf-8")

    if result.returncode != 0:
        return "CAUGHT", mutation.test
    return "MISSED", mutation.test


def main() -> int:
    outcomes: list[tuple[str, str]] = []
    for mutation in MUTATIONS:
        status, detail = _run_mutation(mutation)
        outcomes.append((status, detail))
        print(f"{status}: {detail}")

    missed = [d for s, d in outcomes if s == "MISSED"]
    skipped = [d for s, d in outcomes if s == "SKIP"]
    if missed:
        print(f"\n{len(missed)} mutation(s) were NOT caught — strengthen those tests.")
        return 1
    if skipped:
        print(f"\n{len(skipped)} mutation(s) skipped — update patterns in mutation_smoke.py.")
        return 1
    print(f"\nAll {len(MUTATIONS)} mutations caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
