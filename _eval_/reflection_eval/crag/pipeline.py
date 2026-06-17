"""CRAG eval: ``react`` baseline vs ``react_crag`` on pooled BEIR data.

Primary metrics are context recall / nDCG from qrels. CRAG-specific
``signal_metrics`` track how often the subgraph requeries, trims, or skips.
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

CRAG_PATTERN = "react_crag"
RESULT_PREFIX = "reflection_crag"

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
class CragEvalResult:
    """Outcome of a CRAG vs baseline run."""

    baseline: AgentPatternResult
    pattern: AgentPatternResult
    signal_metrics: dict[str, float]
    deltas: dict[str, float]


def _crag_signal_metrics(per_query: list[AgentQueryScore]) -> dict[str, float]:
    return {
        "crag_use_rate": signal_rate(per_query, "crag_action", "use"),
        "crag_requery_rate": signal_rate(per_query, "crag_action", "requery"),
        "crag_degrade_rate": signal_rate(per_query, "crag_action", "degrade"),
        "crag_web_fallback_rate": signal_rate(per_query, "crag_action", "web_fallback"),
        "crag_verdict_correct_rate": signal_rate(per_query, "crag_verdict", "correct"),
        "crag_verdict_incorrect_rate": signal_rate(per_query, "crag_verdict", "incorrect"),
        "crag_verdict_ambiguous_rate": signal_rate(per_query, "crag_verdict", "ambiguous"),
        "crag_skipped_rate": signal_rate(per_query, "crag_verdict", "skipped"),
    }


async def evaluate_crag(cfg: AgentRunConfig) -> tuple[CragEvalResult, RAGIndexer]:
    """Run baseline + CRAG on one pooled index and aggregate pattern signals."""
    baseline, pattern, indexer = await run_baseline_and_pattern(
        cfg,
        baseline_pattern=BASELINE_PATTERN,
        pattern=CRAG_PATTERN,
    )
    signal_metrics = _crag_signal_metrics(pattern.per_query)
    deltas = metric_deltas(pattern.mean_metrics, baseline.mean_metrics, _DELTA_KEYS)
    return (
        CragEvalResult(
            baseline=baseline,
            pattern=pattern,
            signal_metrics=signal_metrics,
            deltas=deltas,
        ),
        indexer,
    )


def format_crag_table(result: CragEvalResult) -> str:
    """Render a compact markdown summary for terminal output."""
    lines = [
        "| row | queries | ctx_recall@10 | ctx_ndcg@10 | num_rag_calls |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, row in (
        ("baseline", result.baseline),
        ("react_crag", result.pattern),
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
    lines.append("**CRAG signals**")
    for key, value in sorted(result.signal_metrics.items()):
        lines.append(f"- {key}: {value:.3f}")

    lines.append("")
    lines.append("**Deltas (react_crag - react)**")
    for key, value in sorted(result.deltas.items()):
        lines.append(f"- {key}: {value:+.3f}")
    return "\n".join(lines)
