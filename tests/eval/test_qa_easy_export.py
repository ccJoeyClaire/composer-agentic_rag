"""Tests for Easy Dataset export loading (no LLM)."""

from __future__ import annotations

from pathlib import Path

from _eval_.qa_eval.easy_export import assign_query_ids, load_easy_export, slugify_doc_id

_REPO = Path(__file__).resolve().parents[2]
_EXPORT = (
    _REPO
    / "_eval_"
    / "datasets"
    / "Easy-Dataset"
    / "eval-datasets-1781929888288.jsonl"
)


def test_load_easy_export_has_rows() -> None:
    rows = load_easy_export(_EXPORT)
    assert len(rows) == 27
    assert rows[0]["questionType"] == "open_ended"


def test_assign_query_ids_stable() -> None:
    rows = load_easy_export(_EXPORT)
    doc_id = slugify_doc_id("工程技术：在智能体优先的世界中利用 Codex")
    keyed = assign_query_ids(rows, doc_id=doc_id)
    assert keyed[0][0] == f"{doc_id}::q0000"
    assert keyed[-1][0] == f"{doc_id}::q0026"
