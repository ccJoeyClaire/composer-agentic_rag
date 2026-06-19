from __future__ import annotations

import pytest

from tests.conftest import make_chunk
from tests.fakes.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_in_memory_alist_chunks_filter_and_offset() -> None:
    store = InMemoryVectorStore()
    await store.aadd_chunks(
        [
            make_chunk("a", metadata={"doc_id": "d1", "source": "s1"}),
            make_chunk("b", metadata={"doc_id": "d1", "source": "s2"}),
            make_chunk("c", metadata={"doc_id": "d2", "source": "s1"}),
        ],
        [[1.0], [1.0], [1.0]],
    )

    page, next_offset = await store.alist_chunks(limit=1, doc_id="d1")
    assert len(page) == 1
    assert page[0].content == "a"
    assert next_offset == "1"

    page2, done = await store.alist_chunks(limit=5, offset="1", doc_id="d1")
    assert len(page2) == 1
    assert page2[0].content == "b"
    assert done is None

    filtered_source, _ = await store.alist_chunks(limit=10, source="s1")
    assert {chunk.content for chunk in filtered_source} == {"a", "c"}
