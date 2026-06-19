"""Load and index RAG eval run JSON written by :mod:`_eval_.rag_eval.run`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from _eval_.data_preparing.beir import QueryId


class QueryScore(TypedDict):
    """Per-query retrieval metrics for one profile."""

    query_id: str
    num_gold: int
    num_ranked_docs: int
    metrics: dict[str, float]


class ProfileResult(TypedDict):
    """Aggregate and per-query outcome for one RAG profile."""

    profile_id: str
    collection: str
    num_docs_indexed: int
    num_queries: int
    mean_metrics: dict[str, float]
    per_query: list[QueryScore]


class RunConfigSnapshot(TypedDict, total=False):
    """Subset of run config persisted in the results JSON."""

    profiles: list[str]
    k_values: list[int]
    query_limit: int | None
    rel_threshold: int
    max_distractors_per_query: int | None
    predict_question_max_concurrency: int
    chunk_fetch_multiplier: int
    recreate: bool


class RunResult(TypedDict):
    """Top-level artifact from one eval run."""

    dataset: str
    timestamp_utc: str
    config: RunConfigSnapshot
    results: list[ProfileResult]


def load_run_result(path: Path | str) -> RunResult:
    """Parse a run JSON file into a typed :class:`RunResult`.

    Args:
        path: Path to ``*_eval_/results/{dataset}_{timestamp}.json``.

    Returns:
        Parsed run with profile-level and per-query metrics.

    Raises:
        FileNotFoundError: Missing file.
        ValueError: JSON shape does not match the expected run artifact.
    """
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"run JSON root must be an object, got {type(raw).__name__}")

    dataset = raw.get("dataset")
    timestamp_utc = raw.get("timestamp_utc")
    config = raw.get("config")
    results = raw.get("results")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("run JSON missing non-empty 'dataset'")
    if not isinstance(timestamp_utc, str):
        raise ValueError("run JSON missing 'timestamp_utc'")
    if not isinstance(config, dict):
        raise ValueError("run JSON missing 'config' object")
    if not isinstance(results, list) or not results:
        raise ValueError("run JSON missing non-empty 'results' list")

    profiles: list[ProfileResult] = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"results[{idx}] must be an object")
        profile_id = item.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError(f"results[{idx}] missing 'profile_id'")
        per_query_raw = item.get("per_query")
        if not isinstance(per_query_raw, list):
            raise ValueError(f"results[{idx}] missing 'per_query' list")

        per_query: list[QueryScore] = []
        for qidx, row in enumerate(per_query_raw):
            if not isinstance(row, dict):
                raise ValueError(f"results[{idx}].per_query[{qidx}] must be an object")
            query_id = row.get("query_id")
            metrics = row.get("metrics")
            if not isinstance(query_id, str):
                raise ValueError(
                    f"results[{idx}].per_query[{qidx}] missing 'query_id'"
                )
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"results[{idx}].per_query[{qidx}] missing 'metrics'"
                )
            per_query.append(
                QueryScore(
                    query_id=query_id,
                    num_gold=int(row.get("num_gold", 0)),
                    num_ranked_docs=int(row.get("num_ranked_docs", 0)),
                    metrics={str(k): float(v) for k, v in metrics.items()},
                )
            )

        mean_metrics_raw = item.get("mean_metrics")
        if not isinstance(mean_metrics_raw, dict):
            raise ValueError(f"results[{idx}] missing 'mean_metrics'")
        profiles.append(
            ProfileResult(
                profile_id=profile_id,
                collection=str(item.get("collection", "")),
                num_docs_indexed=int(item.get("num_docs_indexed", 0)),
                num_queries=int(item.get("num_queries", len(per_query))),
                mean_metrics={
                    str(k): float(v) for k, v in mean_metrics_raw.items()
                },
                per_query=per_query,
            )
        )

    return RunResult(
        dataset=dataset,
        timestamp_utc=timestamp_utc,
        config=RunConfigSnapshot(**config),
        results=profiles,
    )


def profile_map(run: RunResult) -> dict[str, ProfileResult]:
    """Map ``profile_id`` → profile result; raises on duplicate ids."""
    mapped: dict[str, ProfileResult] = {}
    for profile in run["results"]:
        pid = profile["profile_id"]
        if pid in mapped:
            raise ValueError(f"duplicate profile_id in run: {pid!r}")
        mapped[pid] = profile
    return mapped


def metric_names(run: RunResult) -> list[str]:
    """Return sorted metric keys from the first profile (all profiles share keys)."""
    if not run["results"]:
        return []
    return sorted(run["results"][0]["mean_metrics"].keys())


def per_query_metrics(profile: ProfileResult, metric: str) -> dict[QueryId, float]:
    """Build ``query_id → metric value`` for one profile and metric name."""
    series: dict[QueryId, float] = {}
    for row in profile["per_query"]:
        value = row["metrics"].get(metric)
        if value is None:
            raise KeyError(
                f"metric {metric!r} missing for query {row['query_id']!r} "
                f"in profile {profile['profile_id']!r}"
            )
        series[row["query_id"]] = float(value)
    return series


def common_query_ids(run: RunResult, *profile_ids: str) -> list[QueryId]:
    """Intersection of query ids across the given profiles, stable sorted order."""
    ids = profile_ids or tuple(p["profile_id"] for p in run["results"])
    if not ids:
        return []

    by_profile = profile_map(run)
    sets = [{row["query_id"] for row in by_profile[pid]["per_query"]} for pid in ids]
    common = set.intersection(*sets)
    return sorted(common, key=lambda q: (len(q), q))
