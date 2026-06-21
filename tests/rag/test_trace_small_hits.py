"""Tests for small-to-big trace capture (scheme A)."""

from __future__ import annotations

import pytest

from rag.base import TRACE_SMALL_RETRIEVED_KEY, BaseRetriever, Chunk
from rag.core import RAGRetriever
from rag.document_augmentation.parent_builder import CHUNK_ID_KEY
from rag.retriever.small_to_big_retriever import SmallToBigRetriever
from tests.conftest import make_chunk

pytestmark = pytest.mark.unit


class _FixedHitsRetriever(BaseRetriever):
    def __init__(self, hits: list[Chunk]) -> None:
        self._hits = hits

    async def aretrieve(self, query: str, top_k: int) -> list[Chunk]:
        _ = query
        return self._hits[:top_k]


@pytest.mark.asyncio
async def test_small_to_big_retriever_records_last_small_hits() -> None:
    hits = [make_chunk("small text", metadata={CHUNK_ID_KEY: "doc.md::0"})]
    inner = _FixedHitsRetriever(hits)
    retriever = SmallToBigRetriever(inner, store=None)

    await retriever.aretrieve("query", top_k=1)

    assert len(retriever.last_small_hits) == 1
    assert retriever.last_small_hits[0].content == "small text"


@pytest.mark.asyncio
async def test_aquery_trace_populates_trace_small_retrieved() -> None:
    hits = [make_chunk("small text", metadata={CHUNK_ID_KEY: "doc.md::0"})]
    inner = _FixedHitsRetriever(hits)
    s2b = SmallToBigRetriever(inner, store=None)
    pipeline = RAGRetriever(retriever=s2b, recall_n=5)

    result = await pipeline.aquery_trace("query", top_k=1)

    assert TRACE_SMALL_RETRIEVED_KEY in result.metadata
    assert len(result.metadata[TRACE_SMALL_RETRIEVED_KEY]) == 1
    assert result.metadata[TRACE_SMALL_RETRIEVED_KEY][0].content == "small text"
