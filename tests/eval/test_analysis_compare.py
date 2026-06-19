"""Tests for _eval_.analysis load + compare helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from _eval_.analysis.compare import (
    delta_per_query,
    mean_delta,
    mean_delta_table,
    paired_ttest,
    winrate,
    winrate_matrix,
)
from _eval_.analysis.load_results import (
    common_query_ids,
    load_run_result,
    metric_names,
    profile_map,
)
from _eval_.paths import REPO_ROOT

_NFCORPUS_RESULT = (
    REPO_ROOT / "_eval_" / "results" / "nfcorpus_20260618T045954.json"
)


def _synthetic_run() -> dict:
    return {
        "dataset": "smoke",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "config": {"profiles": ["a", "b"], "k_values": [3, 10]},
        "results": [
            {
                "profile_id": "a",
                "collection": "c_a",
                "num_docs_indexed": 10,
                "num_queries": 2,
                "mean_metrics": {"ndcg@10": 0.5},
                "per_query": [
                    {
                        "query_id": "q1",
                        "num_gold": 1,
                        "num_ranked_docs": 3,
                        "metrics": {"ndcg@10": 0.4},
                    },
                    {
                        "query_id": "q2",
                        "num_gold": 1,
                        "num_ranked_docs": 3,
                        "metrics": {"ndcg@10": 0.6},
                    },
                ],
            },
            {
                "profile_id": "b",
                "collection": "c_b",
                "num_docs_indexed": 10,
                "num_queries": 2,
                "mean_metrics": {"ndcg@10": 0.55},
                "per_query": [
                    {
                        "query_id": "q1",
                        "num_gold": 1,
                        "num_ranked_docs": 3,
                        "metrics": {"ndcg@10": 0.5},
                    },
                    {
                        "query_id": "q2",
                        "num_gold": 1,
                        "num_ranked_docs": 3,
                        "metrics": {"ndcg@10": 0.6},
                    },
                ],
            },
        ],
    }


def test_load_run_result_from_dict_file(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        __import__("json").dumps(_synthetic_run()),
        encoding="utf-8",
    )
    run = load_run_result(path)
    assert run["dataset"] == "smoke"
    assert len(run["results"]) == 2
    assert profile_map(run)["a"]["mean_metrics"]["ndcg@10"] == 0.5


def test_winrate_counts_wins_losses_ties(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        __import__("json").dumps(_synthetic_run()),
        encoding="utf-8",
    )
    run = load_run_result(path)
    result = winrate(
        run,
        reference_profile="a",
        challenger_profile="b",
        metric="ndcg@10",
    )
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 0
    assert result["n"] == 2
    assert result["winrate"] == 0.75


def test_delta_and_mean_delta(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        __import__("json").dumps(_synthetic_run()),
        encoding="utf-8",
    )
    run = load_run_result(path)
    deltas = delta_per_query(
        run,
        reference_profile="a",
        challenger_profile="b",
        metric="ndcg@10",
    )
    assert deltas == {"q1": pytest.approx(0.1), "q2": pytest.approx(0.0)}
    assert mean_delta(
        run,
        reference_profile="a",
        challenger_profile="b",
        metric="ndcg@10",
    ) == pytest.approx(0.05)


def test_mean_delta_table_excludes_reference(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        __import__("json").dumps(_synthetic_run()),
        encoding="utf-8",
    )
    run = load_run_result(path)
    table = mean_delta_table(
        run,
        reference_profile="a",
        profiles=["a", "b"],
        metrics=["ndcg@10"],
    )
    assert set(table.keys()) == {"b"}
    assert table["b"]["ndcg@10"] == pytest.approx(0.05)


def test_paired_ttest_returns_summary(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        __import__("json").dumps(_synthetic_run()),
        encoding="utf-8",
    )
    run = load_run_result(path)
    result = paired_ttest(
        run,
        reference_profile="a",
        challenger_profile="b",
        metric="ndcg@10",
    )
    assert result["n"] == 2
    assert result["mean_delta"] == pytest.approx(0.05)


@pytest.mark.skipif(not _NFCORPUS_RESULT.is_file(), reason="nfcorpus result not present")
def test_load_real_nfcorpus_run() -> None:
    run = load_run_result(_NFCORPUS_RESULT)
    profiles = profile_map(run)
    assert "baseline" in profiles
    assert "token" in profiles
    assert len(common_query_ids(run, "token", "baseline")) == 50
    metrics = metric_names(run)
    assert "ndcg@10" in metrics

    matrix = winrate_matrix(
        run,
        reference_profile="baseline",
        metric="ndcg@10",
    )
    assert "token" in matrix
    assert matrix["token"]["n"] == 50
