"""Tests for pooled-subset selection in _eval_.data_preparing.pooling."""

from __future__ import annotations

from _eval_.data_preparing.pooling import (
    PoolSpec,
    build_pool,
    gold_docs,
    queries_with_gold,
)


def test_gold_docs_respects_threshold() -> None:
    relevance = {"a": 2, "b": 1, "c": 0}
    assert gold_docs(relevance, rel_threshold=2) == {"a"}
    assert gold_docs(relevance, rel_threshold=1) == {"a", "b"}


def test_build_pool_unions_gold_and_capped_distractors() -> None:
    qrels = {
        "q1": {"g1": 2, "g2": 2, "d1": 0, "d2": 0, "d3": 0},
    }
    spec = PoolSpec(rel_threshold=1, max_distractors_per_query=1)
    pool = build_pool(qrels, ["q1"], spec)
    assert pool == {"g1", "g2", "d1"}


def test_queries_with_gold_sorted_stably() -> None:
    qrels = {
        "10": {"a": 1},
        "2": {"b": 1},
        "1": {"c": 0},
    }
    assert queries_with_gold(qrels, rel_threshold=1) == ["2", "10"]
