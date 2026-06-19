from __future__ import annotations

import pytest

from _eval_.analysis.replay.inspect_collection import (
    InspectCollectionRequest,
    inspect_collection,
    summarize_chunks,
)
from rag.base import Chunk
from rag.document_augmentation.parent_builder import CHUNK_ROLE_KEY
from tests.conftest import make_chunk
from tests.fakes.vector_store import InMemoryVectorStore


def test_summarize_chunks_empty() -> None:
    summary = summarize_chunks([])
    assert summary["total_listed"] == 0
    assert summary["chunks_per_doc"] == {}
    assert summary["heading_path_nonempty_ratio"] == 0.0


def test_summarize_chunks_counts_roles_and_lengths() -> None:
    chunks = [
        make_chunk(
            "short",
            metadata={
                "doc_id": "d1",
                "heading_path": "Intro",
                CHUNK_ROLE_KEY: "small",
                "embed_text": "embed-short",
            },
        ),
        make_chunk(
            "longer content here",
            metadata={
                "doc_id": "d1",
                "heading_path": "",
                CHUNK_ROLE_KEY: "parent",
            },
        ),
        make_chunk("other doc", metadata={"doc_id": "d2", "heading_path": "Sec"}),
    ]

    summary = summarize_chunks(chunks)

    assert summary["total_listed"] == 3
    assert summary["chunks_per_doc"] == {"d1": 2, "d2": 1}
    assert summary["heading_path_nonempty_ratio"] == pytest.approx(2 / 3)
    assert summary["chunk_role_counts"] == {"small": 1, "parent": 1}
    assert summary["avg_content_len"] > 0
    assert summary["avg_embed_text_len"] > summary["avg_content_len"] / 2


@pytest.mark.asyncio
async def test_inspect_collection_with_explicit_request() -> None:
    store = InMemoryVectorStore()
    await store.aadd_chunks(
        [
            make_chunk("alpha", metadata={"doc_id": "d1", "source": "s1"}),
            make_chunk("beta", metadata={"doc_id": "d2", "source": "s1"}),
        ],
        [[1.0], [1.0]],
    )

    result = await inspect_collection(
        InspectCollectionRequest(collection="ignored", limit=1),
        store=store,
    )

    assert result["collection"] == "ignored"
    assert result["summary"]["total_listed"] == 1
    assert result["chunks"][0]["content"] == "alpha"
    assert result["next_offset"] == "1"
