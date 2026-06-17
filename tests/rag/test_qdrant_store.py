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
