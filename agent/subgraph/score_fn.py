"""CRAG passage scoring via cross-encoder relevance (independent of reranker wiring).

Run (from repo root):
  python -m agent.subgraph.score_fn --unit
  python -m agent.subgraph.score_fn --query "What is RAG?" --passages "..." "..."
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, List

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 32

ScorePassagesFn = Callable[[str, List[str]], Awaitable[List[dict]]]


@dataclass(frozen=True)
class CrossEncoderScoreConfig:
    """Thresholds and model knobs for mapping cross-encoder scores to CRAG labels.

    Defaults are a starting point; calibrate on dev queries for your model.
    """

    model_name: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    correct_threshold: float = 0.5
    incorrect_threshold: float = 0.0
    device: str | None = None


def score_to_label(
    score: float,
    *,
    correct_threshold: float,
    incorrect_threshold: float,
) -> str:
    """Map one cross-encoder score to a CRAG passage label."""
    if score >= correct_threshold:
        return "correct"
    if score < incorrect_threshold:
        return "incorrect"
    return "ambiguous"


def scores_to_labels(
    scores: List[float],
    *,
    config: CrossEncoderScoreConfig,
) -> List[dict]:
    """Build CRAG label dicts (index, label, score) from raw model scores."""
    return [
        {
            "index": i,
            "label": score_to_label(
                score,
                correct_threshold=config.correct_threshold,
                incorrect_threshold=config.incorrect_threshold,
            ),
            "score": score,
        }
        for i, score in enumerate(scores)
    ]


class _CrossEncoderScorer:
    """Lazy-loaded cross-encoder; separate from ``CrossEncoderReranker`` retrieval path."""

    def __init__(self, config: CrossEncoderScoreConfig) -> None:
        self._config = config
        self._model: object | None = None

    def _model_name(self) -> str:
        return self._config.model_name or os.environ.get(
            "CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL
        )

    def _batch_size(self) -> int:
        env_bs = os.environ.get("CROSS_ENCODER_BATCH_SIZE")
        if self._config.batch_size != DEFAULT_BATCH_SIZE:
            return max(1, self._config.batch_size)
        return max(1, int(env_bs)) if env_bs else DEFAULT_BATCH_SIZE

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name(), device=self._config.device)
        return self._model

    def predict_scores(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        model = self._get_model()
        pairs = [(query, passage) for passage in passages]
        raw = model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=self._batch_size(),
            show_progress_bar=False,
        )
        return [float(score) for score in raw]


def build_cross_encoder_score_fn(
    config: CrossEncoderScoreConfig | None = None,
) -> ScorePassagesFn:
    """Return an async scorer suitable for :class:`~agent.subgraph.CRAG.CragConfig.score_fn`."""
    cfg = config or CrossEncoderScoreConfig()
    scorer = _CrossEncoderScorer(cfg)

    async def score(query: str, passages: List[str]) -> List[dict]:
        if not passages:
            return []
        if not query.strip():
            return [{"index": i, "label": "ambiguous"} for i in range(len(passages))]
        scores = await asyncio.to_thread(scorer.predict_scores, query, passages)
        return scores_to_labels(scores, config=cfg)

    return score


def _run_unit_demo() -> None:
    """Offline smoke: threshold mapping without model download."""
    cfg = CrossEncoderScoreConfig()
    samples = [0.8, 0.3, -0.1]
    print("=== score_to_label (threshold table) ===")
    for score in samples:
        label = score_to_label(
            score,
            correct_threshold=cfg.correct_threshold,
            incorrect_threshold=cfg.incorrect_threshold,
        )
        print(f"  score={score:+.1f} -> {label}")


async def _run_live_demo(query: str, passages: list[str]) -> None:
    score_fn = build_cross_encoder_score_fn()
    labels = await score_fn(query, passages)
    print(f"Query: {query}")
    for item in labels:
        print(
            f"  [{item['index']}] label={item['label']} score={item.get('score')}"
        )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CRAG cross-encoder scoring demo.")
    parser.add_argument("--unit", action="store_true", help="Threshold table only (no model)")
    parser.add_argument("--query", default="What is RAG?")
    parser.add_argument("--passages", nargs="*", default=[
        "Retrieval-augmented generation combines retrieval with LLMs.",
        "The weather in Paris is sunny today.",
    ])
    args = parser.parse_args()

    if args.unit:
        _run_unit_demo()
    else:
        asyncio.run(_run_live_demo(args.query, args.passages))


if __name__ == "__main__":
    _main()
