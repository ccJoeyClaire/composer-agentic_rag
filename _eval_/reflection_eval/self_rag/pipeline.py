"""Self-RAG eval: ``react`` baseline vs ``react_self_rag`` on pooled BEIR data.

Primary metrics are context recall / nDCG from qrels. Self-RAG-specific
``signal_metrics`` track retrieve hints, grounded checks, and retry pressure.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.core import RAGIndexer

from _eval_.config import AgentRunConfig
from _eval_.reflection_eval.beir_runner import run_baseline_and_pattern
from _eval_.reflection_eval.shared import (
    BASELINE_PATTERN,
    AgentPatternResult,
    AgentQueryScore,
    metric_deltas,
    signal_rate,
)

SELF_RAG_PATTERN = "react_self_rag"
RESULT_PREFIX = "reflection_self_rag"

_DELTA_KEYS = (
    "ctx_recall@3",
    "ctx_recall@10",
    "ctx_recall@20",
    "ctx_ndcg@10",
    "ctx_mrr@20",
    "num_rag_calls",
    "num_turns",
)


@dataclass(frozen=True)
class SelfRagEvalResult:
    """Outcome of a Self-RAG vs baseline run."""

    baseline: AgentPatternResult
    pattern: AgentPatternResult
    signal_metrics: dict[str, float]
    deltas: dict[str, float]


def _self_rag_signal_metrics(per_query: list[AgentQueryScore]) -> dict[str, float]:
    return {
        "need_retrieve_rate": signal_rate(per_query, "self_rag_need_retrieve", True),
        "skip_retrieve_rate": signal_rate(per_query, "self_rag_need_retrieve", False),
        "grounded_rate": signal_rate(per_query, "self_rag_grounded", True),
        "ungrounded_rate": signal_rate(per_query, "self_rag_grounded", False),
        "grounded_unknown_rate": signal_rate(per_query, "self_rag_grounded", None),
    }


async def evaluate_self_rag(cfg: AgentRunConfig) -> tuple[SelfRagEvalResult, RAGIndexer]:
    """Run baseline + Self-RAG on one pooled index and aggregate pattern signals."""
    baseline, pattern, indexer = await run_baseline_and_pattern(
        cfg,
        baseline_pattern=BASELINE_PATTERN,
        pattern=SELF_RAG_PATTERN,
    )
    signal_metrics = _self_rag_signal_metrics(pattern.per_query)
    deltas = metric_deltas(pattern.mean_metrics, baseline.mean_metrics, _DELTA_KEYS)
    return (
        SelfRagEvalResult(
            baseline=baseline,
            pattern=pattern,
            signal_metrics=signal_metrics,
            deltas=deltas,
        ),
        indexer,
    )


def format_self_rag_table(result: SelfRagEvalResult) -> str:
    """Render a compact markdown summary for terminal output."""
    lines = [
        "| row | queries | ctx_recall@10 | ctx_ndcg@10 | num_rag_calls |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, row in (
        ("baseline", result.baseline),
        ("react_self_rag", result.pattern),
    ):
        m = row.mean_metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row.num_queries),
                    f"{m.get('ctx_recall@10', 0.0):.3f}",
                    f"{m.get('ctx_ndcg@10', 0.0):.3f}",
                    f"{m.get('num_rag_calls', 0.0):.3f}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("**Self-RAG signals**")
    for key, value in sorted(result.signal_metrics.items()):
        lines.append(f"- {key}: {value:.3f}")

    lines.append("")
    lines.append("**Deltas (react_self_rag - react)**")
    for key, value in sorted(result.deltas.items()):
        lines.append(f"- {key}: {value:+.3f}")
    return "\n".join(lines)
