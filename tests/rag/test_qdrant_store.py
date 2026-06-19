from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from rag.store.qdrant_store import QdrantVectorStore


@pytest.mark.asyncio
async def test_ensure_collection_tolerates_concurrent_create_race() -> None:
    store = QdrantVectorStore(collection="race_test", url="http://127.0.0.1:6333")
    store.client.collection_exists = AsyncMock(return_value=False)
    store.client.create_collection = AsyncMock(
        side_effect=UnexpectedResponse(
            status_code=409,
            reason_phrase="Conflict",
            content=b'{"status":{"error":"already exists"}}',
            headers={},
        )
    )

    await asyncio.gather(
        store.ensure_collection(vector_size=4),
        store.ensure_collection(vector_size=4),
    )

    assert store.client.create_collection.await_count >= 1


@pytest.mark.asyncio
async def test_ensure_collection_serializes_parallel_creates() -> None:
    store = QdrantVectorStore(collection="serialize_test", url="http://127.0.0.1:6333")
    exists = False

    async def _collection_exists(_name: str) -> bool:
        return exists

    async def _create_collection(**_kwargs: object) -> None:
        nonlocal exists
        exists = True

    store.client.collection_exists = AsyncMock(side_effect=_collection_exists)
    store.client.create_collection = AsyncMock(side_effect=_create_collection)

    await asyncio.gather(
        store.ensure_collection(vector_size=4),
        store.ensure_collection(vector_size=4),
        store.ensure_collection(vector_size=4),
    )

    store.client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_alist_chunks_applies_filters_and_pagination() -> None:
    store = QdrantVectorStore(collection="scroll_test", url="http://127.0.0.1:6333")
    store.client.collection_exists = AsyncMock(return_value=True)

    records = [
        type("Rec", (), {"payload": {"content": "a", "metadata": {"doc_id": "d1", "source": "s1"}}})(),
        type("Rec", (), {"payload": {"content": "b", "metadata": {"doc_id": "d2", "source": "s1"}}})(),
    ]

    async def _scroll(**kwargs: object) -> tuple[list[object], str | None]:
        assert kwargs["limit"] == 1
        if kwargs.get("offset") is None:
            return [records[0]], "page-2"
        return [records[1]], None

    store.client.scroll = AsyncMock(side_effect=_scroll)

    page1, next_offset = await store.alist_chunks(limit=1, doc_id="d1", source="s1")
    assert len(page1) == 1
    assert page1[0].content == "a"
    assert next_offset == "page-2"

    page2, done = await store.alist_chunks(limit=1, offset="page-2", doc_id="d1", source="s1")
    assert len(page2) == 1
    assert page2[0].content == "b"
    assert done is None

    scroll_filter = store.client.scroll.await_args_list[0].kwargs["scroll_filter"]
    assert scroll_filter is not None
    assert len(scroll_filter.must) == 2
