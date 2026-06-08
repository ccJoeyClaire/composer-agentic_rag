"""
RAG eval framework (M1 skeleton).

Runs against gold_rag.jsonl and computes Recall@k from keyword / heading heuristics.
Full baseline-vs-stack comparison is documented in docs/TESTING_GUIDE.md (Phase M2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

GOLD_PATH = Path(__file__).parent / "gold_rag.jsonl"


def load_gold_cases(path: Path = GOLD_PATH) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def chunk_matches(case: Dict[str, Any], chunk_text: str, meta: dict) -> bool:
    heading_needle = (case.get("expected_heading_contains") or "").strip()
    if heading_needle:
        heading = (meta or {}).get("heading_path") or ""
        if heading_needle.lower() in heading.lower():
            return True

    keywords = case.get("expected_keywords") or []
    haystack = chunk_text.lower()
    meta_text = json.dumps(meta or {}, ensure_ascii=False).lower()
    hits = sum(1 for kw in keywords if kw.lower() in haystack or kw.lower() in meta_text)
    if keywords and hits >= max(1, len(keywords) // 2):
        return True
    return False


def recall_at_k(chunks: Iterable, case: Dict[str, Any], k: int) -> float:
    selected = list(chunks)[:k]
    for chunk in selected:
        text = getattr(chunk, "content", str(chunk))
        meta = getattr(chunk, "metadata", {}) or {}
        if chunk_matches(case, text, meta):
            return 1.0
    return 0.0


def mean_recall_at_k(all_scores: List[float]) -> float:
    if not all_scores:
        return 0.0
    return sum(all_scores) / len(all_scores)


@pytest.mark.eval
@pytest.mark.requires_api
@pytest.mark.skip(reason="M1 skeleton: enable after wiring pipeline + API keys in CI/local eval")
@pytest.mark.asyncio
async def test_rag_gold_recall_at_3():
    """
    Placeholder eval test.

    When enabled:
      1. Index get_start/工程技术：在智能体优先的世界中利用 Codex.md
      2. Run pipeline.aquery for each gold case
      3. Assert mean Recall@3 >= threshold
    """
    from rag.build import build_RAG_indexer, build_RAG_retriever

    indexer = build_RAG_indexer(
        "eval_gold",
        in_memory=True,
        use_contextual=True,
        use_small_to_big=True,
    )
    build_RAG_retriever(
        "eval_gold",
        in_memory=True,
        use_contextual=True,
        use_small_to_big=True,
        store=indexer.store,
        embedder=indexer.embedder,
    )
    # Index once — requires embedding API
    # await indexer.aindex(...)

    scores: List[float] = []
    for case in load_gold_cases():
        # chunks = await pipeline.aquery(case["query"], top_k=3)
        # scores.append(recall_at_k(chunks, case, k=3))
        scores.append(0.0)

    assert mean_recall_at_k(scores) >= 0.0


@pytest.mark.unit
def test_recall_at_k_helper_detects_keyword_hit():
    from rag.base import Chunk

    case = {"expected_keywords": ["AGENTS.md"], "expected_heading_contains": ""}
    chunks = [
        Chunk(content="unrelated", metadata={}),
        Chunk(content="See AGENTS.md for agent rules.", metadata={}),
    ]
    assert recall_at_k(chunks, case, k=2) == 1.0
    assert recall_at_k(chunks, case, k=1) == 0.0


@pytest.mark.unit
def test_load_gold_cases_reads_jsonl():
    cases = load_gold_cases()
    assert len(cases) >= 3
    assert "query" in cases[0]
