"""Map passage scores to retrieval gate verdict metadata."""

from __future__ import annotations

from agent.types import GateVerdict


def compute_gate_verdict(
    passages: list[str],
    scores: list[float],
    *,
    pass_threshold: float,
) -> tuple[GateVerdict, list[str]]:
    """Classify retrieval quality from passage scores.

    Args:
        passages: Parsed RAG passages (may be empty).
        scores: Relevance scores aligned with ``passages``.
        pass_threshold: Minimum top score for ``pass`` (``qwen3-rerank`` uses 0~1).

    Returns:
        Verdict plus human-readable issue strings for the LLM.
    """
    if not passages:
        return "empty", ["RAG tool returned no passages"]

    if len(scores) != len(passages):
        return "low_quality", [
            f"score count ({len(scores)}) does not match passage count ({len(passages)})"
        ]

    max_score = max(scores)
    if max_score >= pass_threshold:
        return "pass", []

    return "low_quality", [
        f"max relevance {max_score:.2f} below threshold {pass_threshold:.2f}"
    ]


def build_passages_summary(passages: list[str], scores: list[float]) -> str:
    """Short summary for agent metadata / LLM context."""
    if not passages:
        return "0 passages"
    max_score = max(scores) if scores else 0.0
    return f"{len(passages)} passage(s), max relevance={max_score:.2f}"
