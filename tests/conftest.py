"""Shared fixtures for RagPipeLine tests (no changes to production modules)."""

from __future__ import annotations

import hashlib
from typing import Dict, List

import pytest

from rag.base import BaseEmbedder, BaseVectorStore, Chunk


def make_chunk(
    content: str,
    *,
    metadata: dict | None = None,
    score: float = 0.0,
) -> Chunk:
    return Chunk(content=content, metadata=dict(metadata or {}), score=score)


def make_small_chunks(
    texts: List[str],
    *,
    source: str = "doc.md",
    heading_path: str | None = "Section A",
) -> List[Chunk]:
    """Build small chunks as if produced by a chunker (before assign_parent_chunks)."""
    chunks: List[Chunk] = []
    offset = 0
    for i, text in enumerate(texts):
        end = offset + len(text)
        meta = {
            "source": source,
            "heading_path": heading_path,
            "chunk_index": i,
            "start": offset,
            "end": end,
        }
        chunks.append(make_chunk(text, metadata=meta))
        offset = end + 2
    return chunks


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder: identical text → identical vector (cosine-friendly)."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vectorize(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(b) / 255.0 for b in digest[: self.dim]]
        if len(raw) < self.dim:
            raw.extend([0.0] * (self.dim - len(raw)))
        return raw

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(t) for t in texts]

    async def aembed_query(self, query: str) -> List[float]:
        return self._vectorize(query)


class InMemoryChunkStore(BaseVectorStore):
    """Minimal store for unit tests of Small-to-Big member resolution."""

    def __init__(self, chunks_by_id: Dict[str, Chunk] | None = None):
        self._by_id = dict(chunks_by_id or {})

    async def aadd_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        for chunk in chunks:
            cid = (chunk.metadata or {}).get("chunk_id")
            if cid:
                self._by_id[cid] = chunk

    async def asearch(self, query_vector: List[float], top_k: int) -> List[Chunk]:
        return []

    async def aretrieve_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        return [self._by_id[cid] for cid in chunk_ids if cid in self._by_id]


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=8)


@pytest.fixture
def sample_markdown() -> str:
    return """# Alpha

First paragraph about retrieval quality and vector search.

Second paragraph still about retrieval and embeddings.

# Beta

Unrelated topic about cooking pasta and tomato sauce.
"""


@pytest.fixture
def in_memory_vector_store():
    """Test-only vector store for pipeline integration (see tests/fakes/)."""
    from tests.fakes.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    yield store
    store.clear()


# Backward-compatible alias used in integration tests.
@pytest.fixture
def in_memory_qdrant_store(in_memory_vector_store):
    return in_memory_vector_store
