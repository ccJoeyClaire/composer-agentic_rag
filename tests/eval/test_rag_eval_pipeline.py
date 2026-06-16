"""Tests for RAG eval pipeline scoring helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from _eval_.config import RunConfig
from _eval_.data_preparing.beir import EvalQuery, SOURCE_META_KEY
from _eval_.data_preparing.pooling import PoolSpec
from _eval_.rag_eval.pipeline import ranked_doc_ids, score_queries
from rag.base import Chunk


def _chunk(doc_id: str) -> Chunk:
    return Chunk(content=f"text-{doc_id}", metadata={SOURCE_META_KEY: doc_id})


def test_ranked_doc_ids_dedupes_preserving_order() -> None:
    chunks = [_chunk("a"), _chunk("a"), _chunk("b"), _chunk("c"), _chunk("b")]
    assert ranked_doc_ids(chunks) == ["a", "b", "c"]


def test_ranked_doc_ids_skips_missing_source() -> None:
    chunks = [Chunk(content="no meta", metadata={}), _chunk("x")]
    assert ranked_doc_ids(chunks) == ["x"]


@pytest.mark.asyncio
async def test_score_queries_computes_metric_keys() -> None:
    cfg = RunConfig(
        dataset="trec-covid",
        profiles=["baseline"],
        pool_spec=PoolSpec(rel_threshold=1),
        k_values=(3,),
    )
    queries = {"q1": EvalQuery(query_id="q1", text="query text")}
    qrels = {"q1": {"gold": 2, "noise": 0}}
    retriever = AsyncMock()
    retriever.aquery.return_value = [_chunk("gold"), _chunk("noise")]

    scores = await score_queries(retriever, queries, qrels, ["q1"], cfg)
    assert len(scores) == 1
    row = scores[0]
    assert row["query_id"] == "q1"
    assert row["num_gold"] == 1
    assert row["num_ranked_docs"] == 2
    assert "recall@3" in row["metrics"]
    assert "ndcg@3" in row["metrics"]
    assert "hit@3" in row["metrics"]
    assert "mrr@3" in row["metrics"]
