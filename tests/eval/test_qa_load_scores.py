"""Tests for QA rubric score loading helpers (no plotting)."""

from __future__ import annotations

from pathlib import Path

from _eval_.qa_eval.load_scores import (
    failure_counts,
    load_score_bundle,
    per_query_delta,
    winrate,
)

_REPO = Path(__file__).resolve().parents[2]
_SCORES = _REPO / "_eval_" / "datasets" / "Easy-Dataset" / "rubric_scores.jsonl"
_GOLD = _REPO / "_eval_" / "datasets" / "Easy-Dataset" / "gold_rubric.jsonl"


def test_load_score_bundle_joins_gold() -> None:
    scores, gold, joined = load_score_bundle(scores_path=_SCORES, gold_path=_GOLD)
    assert len(scores) == 54
    assert len(gold) == 27
    assert len(joined) == 54
    assert joined[0]["question_type"] in {
        "open_ended",
        "short_answer",
        "multiple_choice",
        "single_choice",
        "true_false",
        "unknown",
    }


def test_per_query_delta_and_winrate() -> None:
    scores, _, _ = load_score_bundle(scores_path=_SCORES, gold_path=_GOLD)
    deltas = per_query_delta(
        scores,
        reference_profile="react_baseline",
        challenger_profile="react_crag_baseline",
        metric="correct_rate",
    )
    assert len(deltas) == 27
    stats = winrate(
        scores,
        reference_profile="react_baseline",
        challenger_profile="react_crag_baseline",
        metric="complete_rate",
    )
    assert stats["n"] == 27
    assert 0.0 <= float(stats["winrate"]) <= 1.0


def test_failure_counts_non_empty() -> None:
    scores, _, _ = load_score_bundle(scores_path=_SCORES, gold_path=_GOLD)
    failures = failure_counts(scores, profile="react_baseline", top_n=5)
    assert failures
    assert failures[0].fail_count >= failures[-1].fail_count
