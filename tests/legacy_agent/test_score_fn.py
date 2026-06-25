"""Unit tests for CRAG cross-encoder passage scoring."""

from __future__ import annotations

from legacy.agent.subgraph.score_fn import (
    CrossEncoderScoreConfig,
    score_to_label,
    scores_to_labels,
)

pytestmark = __import__("pytest").mark.unit


def test_score_to_label_thresholds() -> None:
    cfg = CrossEncoderScoreConfig(correct_threshold=0.5, incorrect_threshold=0.0)
    assert score_to_label(0.8, correct_threshold=cfg.correct_threshold, incorrect_threshold=cfg.incorrect_threshold) == "correct"
    assert score_to_label(0.5, correct_threshold=cfg.correct_threshold, incorrect_threshold=cfg.incorrect_threshold) == "correct"
    assert score_to_label(0.25, correct_threshold=cfg.correct_threshold, incorrect_threshold=cfg.incorrect_threshold) == "ambiguous"
    assert score_to_label(-0.1, correct_threshold=cfg.correct_threshold, incorrect_threshold=cfg.incorrect_threshold) == "incorrect"


def test_scores_to_labels_preserves_index_and_score() -> None:
    cfg = CrossEncoderScoreConfig(correct_threshold=1.0, incorrect_threshold=0.0)
    labels = scores_to_labels([1.5, 0.5, -1.0], config=cfg)
    assert labels == [
        {"index": 0, "label": "correct", "score": 1.5},
        {"index": 1, "label": "ambiguous", "score": 0.5},
        {"index": 2, "label": "incorrect", "score": -1.0},
    ]
