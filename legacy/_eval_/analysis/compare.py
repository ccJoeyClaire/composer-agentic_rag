"""Paired comparisons between RAG eval profiles (delta, winrate, t-test)."""

from __future__ import annotations

import math
import statistics
from typing import TypedDict

from _eval_.analysis.load_results import RunResult, per_query_metrics, profile_map
from _eval_.data_preparing.beir import QueryId


class WinrateResult(TypedDict):
    """Head-to-head win/tie/loss counts between two profiles on one metric."""

    wins: int
    losses: int
    ties: int
    n: int
    winrate: float


class PairedTTestResult(TypedDict):
    """Paired t-test on per-query metric deltas (challenger − reference)."""

    metric: str
    reference_profile: str
    challenger_profile: str
    n: int
    mean_delta: float
    std_delta: float
    t_statistic: float
    p_value: float | None


def _paired_series(
    run: RunResult,
    *,
    reference_profile: str,
    challenger_profile: str,
    metric: str,
) -> tuple[list[QueryId], list[float], list[float]]:
    profiles = profile_map(run)
    ref_series = per_query_metrics(profiles[reference_profile], metric)
    ch_series = per_query_metrics(profiles[challenger_profile], metric)
    shared = sorted(set(ref_series) & set(ch_series), key=lambda q: (len(q), q))
    if not shared:
        raise ValueError(
            f"no shared queries between {reference_profile!r} and "
            f"{challenger_profile!r}"
        )
    ref_vals = [ref_series[qid] for qid in shared]
    ch_vals = [ch_series[qid] for qid in shared]
    return shared, ref_vals, ch_vals


def delta_per_query(
    run: RunResult,
    *,
    reference_profile: str,
    challenger_profile: str,
    metric: str,
) -> dict[QueryId, float]:
    """Per-query delta: challenger − reference for ``metric``."""
    query_ids, ref_vals, ch_vals = _paired_series(
        run,
        reference_profile=reference_profile,
        challenger_profile=challenger_profile,
        metric=metric,
    )
    return {
        qid: ch_val - ref_val
        for qid, ref_val, ch_val in zip(query_ids, ref_vals, ch_vals)
    }


def mean_delta(
    run: RunResult,
    *,
    reference_profile: str,
    challenger_profile: str,
    metric: str,
) -> float:
    """Mean per-query delta (challenger − reference)."""
    deltas = delta_per_query(
        run,
        reference_profile=reference_profile,
        challenger_profile=challenger_profile,
        metric=metric,
    )
    values = list(deltas.values())
    return statistics.mean(values) if values else 0.0


def winrate(
    run: RunResult,
    *,
    reference_profile: str,
    challenger_profile: str,
    metric: str,
    tie_weight: float = 0.5,
) -> WinrateResult:
    """Fraction of shared queries where challenger beats reference on ``metric``.

    Ties count as ``tie_weight`` (default 0.5) toward the win numerator.
    """
    if not 0.0 <= tie_weight <= 1.0:
        raise ValueError("tie_weight must be in [0, 1]")

    deltas = delta_per_query(
        run,
        reference_profile=reference_profile,
        challenger_profile=challenger_profile,
        metric=metric,
    )
    wins = sum(1 for d in deltas.values() if d > 0)
    losses = sum(1 for d in deltas.values() if d < 0)
    ties = sum(1 for d in deltas.values() if d == 0)
    n = len(deltas)
    score = wins + tie_weight * ties
    return WinrateResult(
        wins=wins,
        losses=losses,
        ties=ties,
        n=n,
        winrate=score / n if n else 0.0,
    )


def winrate_matrix(
    run: RunResult,
    *,
    reference_profile: str,
    profiles: list[str] | None = None,
    metric: str,
    tie_weight: float = 0.5,
) -> dict[str, WinrateResult]:
    """Winrate of each profile vs ``reference_profile`` on one metric."""
    candidates = profiles or [p["profile_id"] for p in run["results"]]
    return {
        pid: winrate(
            run,
            reference_profile=reference_profile,
            challenger_profile=pid,
            metric=metric,
            tie_weight=tie_weight,
        )
        for pid in candidates
        if pid != reference_profile
    }


def mean_delta_table(
    run: RunResult,
    *,
    reference_profile: str,
    profiles: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Mean delta (challenger − reference) for each profile × metric."""
    candidates = profiles or [p["profile_id"] for p in run["results"]]
    metric_list = metrics or sorted(run["results"][0]["mean_metrics"].keys())
    table: dict[str, dict[str, float]] = {}
    for pid in candidates:
        if pid == reference_profile:
            continue
        table[pid] = {
            metric: mean_delta(
                run,
                reference_profile=reference_profile,
                challenger_profile=pid,
                metric=metric,
            )
            for metric in metric_list
        }
    return table


def paired_ttest(
    run: RunResult,
    *,
    reference_profile: str,
    challenger_profile: str,
    metric: str,
) -> PairedTTestResult:
    """Two-sided paired t-test on per-query deltas (challenger − reference).

    ``p_value`` is computed via ``scipy.stats.ttest_rel`` when SciPy is installed;
    otherwise only ``t_statistic`` and summary stats are returned.
    """
    _query_ids, ref_vals, ch_vals = _paired_series(
        run,
        reference_profile=reference_profile,
        challenger_profile=challenger_profile,
        metric=metric,
    )
    deltas = [ch - ref for ref, ch in zip(ref_vals, ch_vals)]
    n = len(deltas)
    mean_d = statistics.mean(deltas)
    std_d = statistics.stdev(deltas) if n > 1 else 0.0

    if n < 2:
        t_stat = 0.0
    elif std_d == 0.0:
        t_stat = math.copysign(float("inf"), mean_d) if mean_d != 0.0 else 0.0
    else:
        t_stat = mean_d / (std_d / math.sqrt(n))

    p_value: float | None = None
    if n >= 2:
        try:
            from scipy.stats import ttest_rel  # type: ignore[import-untyped]

            result = ttest_rel(ch_vals, ref_vals)
            p_value = float(result.pvalue)
        except ImportError:
            p_value = None

    return PairedTTestResult(
        metric=metric,
        reference_profile=reference_profile,
        challenger_profile=challenger_profile,
        n=n,
        mean_delta=mean_d,
        std_delta=std_d,
        t_statistic=t_stat,
        p_value=p_value,
    )
