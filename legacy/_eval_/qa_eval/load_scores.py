"""Load and reshape QA rubric score artifacts for analysis and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _eval_.qa_eval.easy_export import load_gold_rubric
from _eval_.qa_eval.score_rubric import load_score_results, summarize_by_profile
from _eval_.qa_eval.types import RubricGold, RubricScoreResult

MetricName = Literal["correct_rate", "complete_rate"]
RubricBucket = Literal["incident", "key_point"]


@dataclass(frozen=True)
class RubricFailure:
    """One rubric item aggregated across scored answers."""

    bucket: RubricBucket
    item: str
    fail_count: int
    total_count: int

    @property
    def fail_rate(self) -> float:
        return self.fail_count / self.total_count if self.total_count else 0.0


def short_query_label(query_id: str) -> str:
    """Return the stable suffix (e.g. ``q0003``) from a full query id."""
    if "::" in query_id:
        return query_id.rsplit("::", maxsplit=1)[-1]
    return query_id


def join_scores_with_gold(
    scores: list[RubricScoreResult],
    gold_records: list[RubricGold],
) -> list[dict[str, object]]:
    """Attach question metadata from gold rubric to each score row."""
    gold_by_id = {record["query_id"]: record for record in gold_records}
    joined: list[dict[str, object]] = []
    for row in scores:
        gold = gold_by_id.get(row["query_id"], {})
        joined.append(
            {
                "query_id": row["query_id"],
                "query_label": short_query_label(row["query_id"]),
                "profile": row["profile"],
                "question": gold.get("question", ""),
                "question_type": gold.get("question_type", "unknown"),
                "correct_rate": row["correct_rate"],
                "complete_rate": row["complete_rate"],
                "incident_scores": row.get("incident_scores") or [],
                "key_point_scores": row.get("key_point_scores") or [],
            }
        )
    return joined


def per_query_delta(
    scores: list[RubricScoreResult],
    *,
    reference_profile: str,
    challenger_profile: str,
    metric: MetricName,
) -> dict[str, float]:
    """Per-query delta: challenger − reference for one metric."""
    ref = {
        row["query_id"]: row[metric]
        for row in scores
        if row["profile"] == reference_profile
    }
    ch = {
        row["query_id"]: row[metric]
        for row in scores
        if row["profile"] == challenger_profile
    }
    shared = sorted(set(ref) & set(ch), key=short_query_label)
    if not shared:
        raise ValueError(
            f"no shared queries between {reference_profile!r} and {challenger_profile!r}"
        )
    return {qid: ch[qid] - ref[qid] for qid in shared}


def winrate(
    scores: list[RubricScoreResult],
    *,
    reference_profile: str,
    challenger_profile: str,
    metric: MetricName,
) -> dict[str, float | int]:
    """Head-to-head win/tie/loss counts on one metric."""
    deltas = per_query_delta(
        scores,
        reference_profile=reference_profile,
        challenger_profile=challenger_profile,
        metric=metric,
    )
    wins = sum(1 for value in deltas.values() if value > 0)
    losses = sum(1 for value in deltas.values() if value < 0)
    ties = sum(1 for value in deltas.values() if value == 0)
    n = len(deltas)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n": n,
        "winrate": (wins + 0.5 * ties) / n if n else 0.0,
    }


def winrate_matrix(
    scores: list[RubricScoreResult],
    *,
    reference_profile: str,
    profiles: list[str] | None = None,
    metric: MetricName,
) -> dict[str, dict[str, float | int]]:
    """Winrate of each profile vs ``reference_profile`` on one metric."""
    candidates = profiles or sorted({row["profile"] for row in scores})
    return {
        profile: winrate(
            scores,
            reference_profile=reference_profile,
            challenger_profile=profile,
            metric=metric,
        )
        for profile in candidates
        if profile != reference_profile
    }


def mean_delta_table(
    scores: list[RubricScoreResult],
    *,
    reference_profile: str,
    profiles: list[str] | None = None,
    metrics: list[MetricName] | None = None,
) -> dict[str, dict[str, float]]:
    """Mean delta (challenger − reference) for each profile × metric."""
    metric_list: list[MetricName] = metrics or ["correct_rate", "complete_rate"]
    candidates = profiles or sorted({row["profile"] for row in scores})
    table: dict[str, dict[str, float]] = {}
    for profile in candidates:
        if profile == reference_profile:
            continue
        table[profile] = {}
        for metric in metric_list:
            deltas = per_query_delta(
                scores,
                reference_profile=reference_profile,
                challenger_profile=profile,
                metric=metric,
            )
            values = list(deltas.values())
            table[profile][metric] = sum(values) / len(values) if values else 0.0
    return table


def failure_counts(
    scores: list[RubricScoreResult],
    *,
    profile: str | None = None,
    top_n: int = 12,
) -> list[RubricFailure]:
    """Rank rubric items by how often they score 0."""
    totals: dict[tuple[RubricBucket, str], list[int]] = {}
    for row in scores:
        if profile is not None and row["profile"] != profile:
            continue
        for bucket, items in (
            ("incident", row.get("incident_scores") or []),
            ("key_point", row.get("key_point_scores") or []),
        ):
            for item_row in items:
                key = (bucket, item_row["item"])
                totals.setdefault(key, []).append(int(item_row["score"]))

    failures: list[RubricFailure] = []
    for (bucket, item), values in totals.items():
        fail_count = sum(1 for score in values if score == 0)
        if fail_count == 0:
            continue
        failures.append(
            RubricFailure(
                bucket=bucket,
                item=item,
                fail_count=fail_count,
                total_count=len(values),
            )
        )
    failures.sort(key=lambda row: (-row.fail_count, -row.fail_rate, row.item))
    return failures[:top_n]


def load_score_bundle(
    *,
    scores_path: Path,
    gold_path: Path,
) -> tuple[list[RubricScoreResult], list[RubricGold], list[dict[str, object]]]:
    """Load scores + gold and return joined analysis rows."""
    scores = load_score_results(scores_path)
    gold_records = load_gold_rubric(gold_path)
    joined = join_scores_with_gold(scores, gold_records)
    return scores, gold_records, joined
