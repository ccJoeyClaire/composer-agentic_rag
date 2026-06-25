"""Load run metrics and compute deltas between runs.

Used in Step 5 (visualization / comparison) to answer:
  - Does profile X beat token?
  - Does agent pattern Y beat direct RAG?
"""

from __future__ import annotations

from pathlib import Path

from eval.run_config import EvalRunConfig, checker_output_path
from eval.score import build_run_metrics, load_checker_output
from eval.types import MetricDelta, RunMetrics

# Canonical metric keys for delta comparison.
PRIMARY_METRICS: tuple[str, ...] = (
    "overall.f1",
    "overall.precision",
    "overall.recall",
    "retriever.claim_recall",
    "retriever.context_precision",
    "generator.faithfulness",
    "generator.hallucination",
    "generator.context_utilization",
)


def _resolve_metric(metrics: RunMetrics, dotted_key: str) -> float:
    """Resolve a dotted metric key like ``overall.f1`` from a :class:`RunMetrics`.

    Args:
        metrics:    Annotated run metrics.
        dotted_key: Two-part key ``"{section}.{field}"``.

    Returns:
        Metric value; 0.0 when the key is absent.
    """
    section, field = dotted_key.split(".", 1)
    section_dict = metrics.get(section, {})  # type: ignore[literal-required]
    return float(section_dict.get(field, 0.0))  # type: ignore[union-attr]


def load_run_metrics(
    config: EvalRunConfig,
    query_count: int = 0,
    *,
    output_path: Path | None = None,
) -> RunMetrics:
    """Load RAGChecker output for one run and annotate with run identity.

    Args:
        config:      Eval run config.
        query_count: Number of queries evaluated in this run.
        output_path: Override path; defaults to checker output path.

    Returns:
        :class:`RunMetrics` for comparison.
    """
    resolved = output_path if output_path is not None else checker_output_path(config)
    checker_output = load_checker_output(resolved)
    return build_run_metrics(config, checker_output, query_count=query_count)


def compute_deltas(
    baseline: RunMetrics,
    candidate: RunMetrics,
    metrics: tuple[str, ...] = PRIMARY_METRICS,
) -> list[MetricDelta]:
    """Compute absolute deltas for selected metrics.

    Args:
        baseline:  Reference run (e.g. token + direct_rag).
        candidate: Run being evaluated.
        metrics:   Dotted metric keys to compare.

    Returns:
        One :class:`MetricDelta` per metric key.
    """
    deltas: list[MetricDelta] = []
    for key in metrics:
        base_val = _resolve_metric(baseline, key)
        cand_val = _resolve_metric(candidate, key)
        delta = cand_val - base_val
        # For hallucination, lower is better; for everything else, higher is better.
        improved = delta < 0 if key.endswith("hallucination") else delta > 0
        deltas.append(
            MetricDelta(
                metric=key,
                baseline_val=base_val,
                candidate_val=cand_val,
                delta=delta,
                improved=improved,
            )
        )
    return deltas


def load_all_run_metrics(
    matrix: list[EvalRunConfig],
    runs_root: Path | None = None,
) -> list[RunMetrics]:
    """Load metrics for every run in a matrix that has a checker output file.

    Runs without a checker output file are silently skipped so partial
    eval progress can still be compared.

    Args:
        matrix:    Full run matrix.
        runs_root: Override runs root directory.

    Returns:
        Loaded :class:`RunMetrics` in matrix order (missing runs omitted).
    """
    loaded: list[RunMetrics] = []
    for cfg in matrix:
        out_path = (
            runs_root / "checking_outputs" / f"{cfg.run_id}.json"
            if runs_root is not None
            else checker_output_path(cfg)
        )
        if not out_path.exists():
            continue
        checker_output = load_checker_output(out_path)
        loaded.append(build_run_metrics(cfg, checker_output, query_count=0))
    return loaded
