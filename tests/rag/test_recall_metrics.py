"""Unit tests for eval recall helpers (algorithm only, not gold eval)."""

from __future__ import annotations

import pytest

from eval.rag.metrics.recall import load_gold_cases, recall_at_k
from eval.paths import dataset_dir
from rag.base import Chunk

pytestmark = pytest.mark.unit


def test_recall_at_k_helper_detects_keyword_hit():
    case = {"expected_keywords": ["AGENTS.md"], "expected_heading_contains": ""}
    chunks = [
        Chunk(content="unrelated", metadata={}),
        Chunk(content="See AGENTS.md for agent rules.", metadata={}),
    ]
    assert recall_at_k(chunks, case, k=2) == 1.0
    assert recall_at_k(chunks, case, k=1) == 0.0


def test_load_smoke_gold_rag_has_query():
    cases = load_gold_cases(dataset_dir("smoke") / "gold_rag.jsonl")
    assert len(cases) >= 1
    assert "query" in cases[0]
