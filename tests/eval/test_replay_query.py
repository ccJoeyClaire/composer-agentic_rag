from __future__ import annotations

import pytest

from _eval_.analysis.replay.query import parse_stages, resolve_gold_doc_ids, resolve_query_text
from _eval_.analysis.replay.rank_report import gold_rank_report, rank_of_doc
from tests.conftest import make_chunk


def _chunk(doc_id: str, *, score: float = 0.0):
    return make_chunk(f"text-{doc_id}", metadata={"doc_id": doc_id}, score=score)


def test_rank_of_doc_deduplicates_preserving_order() -> None:
    chunks = [_chunk("a"), _chunk("a"), _chunk("b"), _chunk("c")]
    assert rank_of_doc(chunks, "a") == 1
    assert rank_of_doc(chunks, "b") == 2
    assert rank_of_doc(chunks, "missing") is None


def test_gold_rank_report_across_stages() -> None:
    retrieved = [_chunk("x"), _chunk("gold"), _chunk("y")]
    reranked = [_chunk("gold"), _chunk("x"), _chunk("y")]
    final = [_chunk("gold"), _chunk("x")]

    rows = gold_rank_report(
        {"gold", "missing"},
        retrieved=retrieved,
        reranked=reranked,
        final=final,
        top_k=2,
    )
    by_id = {row["doc_id"]: row for row in rows}

    assert by_id["gold"]["rank_retrieved"] == 2
    assert by_id["gold"]["rank_reranked"] == 1
    assert by_id["gold"]["rank_final"] == 1
    assert by_id["gold"]["in_top_k"] is True
    assert by_id["missing"]["rank_retrieved"] is None
    assert by_id["missing"]["in_top_k"] is False


def test_parse_stages() -> None:
    assert parse_stages("retrieved,final") == ("retrieved", "final")


def test_resolve_query_text_requires_input() -> None:
    with pytest.raises(ValueError, match="query text or query_id"):
        resolve_query_text(dataset="nfcorpus", query=None, query_id=None)


def test_resolve_gold_doc_ids_explicit() -> None:
    ids = resolve_gold_doc_ids(
        dataset="nfcorpus",
        query_id=None,
        gold_doc_ids=frozenset({"a", "b"}),
    )
    assert ids == {"a", "b"}
