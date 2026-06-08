"""Tests for rag.retriever.small_to_big_retriever."""

from __future__ import annotations

import pytest

from rag.base import BaseRetriever, Chunk
from rag.document_augmentation.parent_builder import (
    ANCHOR_WINDOW_KEY,
    CHUNK_ID_KEY,
    CHUNK_ROLE_KEY,
    PARENT_CONTENT_KEY,
    assign_parent_chunks,
)
from rag.retriever.small_to_big_retriever import (
    SmallToBigRetriever,
    expand_small_hits_to_parents,
)
from tests.conftest import InMemoryChunkStore, make_chunk, make_small_chunks

pytestmark = pytest.mark.unit


def _indexed_small_hits(texts: list[str], *, source: str = "doc.md") -> list[Chunk]:
    chunks = make_small_chunks(texts, source=source)
    assign_parent_chunks(chunks, parent_token_budget=512)
    hits = []
    for i, c in enumerate(chunks):
        hits.append(
            Chunk(
                content=c.content,
                metadata=dict(c.metadata or {}),
                score=1.0 - i * 0.1,
            )
        )
    return hits


@pytest.mark.asyncio
async def test_expand_single_hit_materializes_parent():
    hits = _indexed_small_hits(["part one", "part two", "part three"])
    target = hits[1]

    parents = await expand_small_hits_to_parents([target], top_k=1, store=None)
    assert len(parents) == 1
    assert parents[0].metadata[CHUNK_ROLE_KEY] == "parent"
    assert PARENT_CONTENT_KEY in parents[0].metadata
    assert "part two" in parents[0].content


@pytest.mark.asyncio
async def test_expand_overlapping_hits_merge_to_one_parent():
    hits = _indexed_small_hits(["a", "b", "c", "d"])
    group = [hits[0], hits[1]]

    parents = await expand_small_hits_to_parents(group, top_k=2, store=None)
    assert len(parents) == 1
    assert parents[0].metadata[CHUNK_ROLE_KEY] == "parent"


@pytest.mark.asyncio
async def test_expand_disjoint_hits_return_multiple_parents():
    chunks = [
        make_chunk("left", metadata={"source": "s.md", "heading_path": "L"}),
        make_chunk("right", metadata={"source": "s.md", "heading_path": "R"}),
    ]
    assign_parent_chunks(chunks, parent_token_budget=512)
    hit_left = Chunk(content=chunks[0].content, metadata=chunks[0].metadata, score=0.9)
    hit_right = Chunk(content=chunks[1].content, metadata=chunks[1].metadata, score=0.8)

    parents = await expand_small_hits_to_parents(
        [hit_left, hit_right], top_k=2, store=None
    )
    assert len(parents) == 2


@pytest.mark.asyncio
async def test_expand_fetches_missing_members_from_store():
    chunks = make_small_chunks(["one", "two", "three"])
    assign_parent_chunks(chunks, parent_token_budget=512)
    by_id = {(c.metadata or {})[CHUNK_ID_KEY]: c for c in chunks}
    store = InMemoryChunkStore(by_id)

    center_hit = Chunk(
        content=chunks[1].content,
        metadata=dict(chunks[1].metadata or {}),
        score=0.95,
    )
    parents = await expand_small_hits_to_parents(
        [center_hit], top_k=1, store=store
    )
    assert len(parents) == 1
    assert "one" in parents[0].content
    assert "three" in parents[0].content


class _StaticRetriever(BaseRetriever):
    def __init__(self, hits: list[Chunk]):
        self.hits = hits

    async def aretrieve(self, query: str, top_k: int) -> list[Chunk]:
        return self.hits[:top_k]


@pytest.mark.asyncio
async def test_small_to_big_retriever_empty_query():
    inner = _StaticRetriever([])
    retriever = SmallToBigRetriever(inner, store=InMemoryChunkStore())
    assert await retriever.aretrieve("  ", top_k=3) == []


@pytest.mark.asyncio
async def test_small_to_big_retriever_recall_multiplier():
    hits = _indexed_small_hits(["x", "y"])
    inner = _StaticRetriever(hits)

    class _SpyRetriever(BaseRetriever):
        last_top_k: int | None = None

        def __init__(self, wrapped: BaseRetriever):
            self.wrapped = wrapped

        async def aretrieve(self, query: str, top_k: int) -> list[Chunk]:
            self.last_top_k = top_k
            return await self.wrapped.aretrieve(query, top_k=top_k)

    spy = _SpyRetriever(inner)
    retriever = SmallToBigRetriever(spy, recall_multiplier=4)
    await retriever.aretrieve("q", top_k=2)
    assert spy.last_top_k == 8
