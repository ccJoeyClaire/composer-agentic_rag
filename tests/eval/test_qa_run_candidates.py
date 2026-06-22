"""Tests for QA candidate runner helpers (no LLM / Qdrant)."""

from __future__ import annotations

from pathlib import Path

from _eval_.qa_eval.easy_export import load_gold_rubric
from _eval_.qa_eval.run_candidates import (
    _resolve_source_bundle,
    _select_records,
    profile_label,
    qa_collection_name,
)

_REPO = Path(__file__).resolve().parents[2]
_GOLD = _REPO / "_eval_" / "datasets" / "Easy-Dataset" / "gold_rubric.jsonl"


def test_qa_collection_name_ascii_safe() -> None:
    name = qa_collection_name("工程技术-在智能体优先的世界中利用-codex", "baseline")
    assert name == "qa_eval_codex_baseline"


def test_profile_label() -> None:
    assert profile_label("react", "baseline") == "react_baseline"


def test_resolve_source_bundle_from_gold() -> None:
    records = load_gold_rubric(_GOLD)
    doc_id, source_path = _resolve_source_bundle(records)
    assert doc_id == "工程技术-在智能体优先的世界中利用-codex"
    assert source_path.is_file()


def test_select_records_limit_and_query_id() -> None:
    records = load_gold_rubric(_GOLD)
    limited = _select_records(records, limit=2, query_ids=None)
    assert len(limited) == 2
    assert limited[0]["query_id"] == records[0]["query_id"]

    target_id = records[0]["query_id"]
    picked = _select_records(
        records,
        limit=None,
        query_ids=frozenset({target_id}),
    )
    assert len(picked) == 1
    assert picked[0]["query_id"] == target_id
