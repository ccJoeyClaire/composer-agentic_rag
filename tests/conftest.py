"""Shared fixtures for RagPipeLine tests (no changes to production modules)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Union

import pytest

from rag.base import BaseEmbedder, BaseVectorStore, Chunk

Chunkish = Union[Chunk, Sequence[Chunk]]


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


def _normalize_chunks(chunks: Chunkish) -> List[Chunk]:
    if isinstance(chunks, Chunk):
        return [chunks]
    return list(chunks)


def _pick_metadata_keys(meta: Mapping[str, Any], only: Sequence[str] | None) -> dict:
    if only is None:
        return dict(meta)
    return {k: meta[k] for k in only if k in meta}


def show_metadata(
    chunks: Chunkish,
    *,
    only: Sequence[str] | None = None,
    content: bool = False,
) -> str:
    """
    Pretty-print chunk metadata for local debugging (pytest -s / REPL).

    only: if set, print only these top-level metadata keys (extra keys are ignored).
    content: include chunk.content in the output.
    """
    rows: List[dict] = []
    for i, chunk in enumerate(_normalize_chunks(chunks)):
        row: dict = {
            "index": i,
            "metadata": _pick_metadata_keys(chunk.metadata or {}, only),
        }
        if content:
            row["content"] = chunk.content
        rows.append(row)

    text = json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False)
    print(text)
    return text


def metadata_diff(
    before: Chunkish | Mapping[str, Any],
    after: Chunkish | Mapping[str, Any],
    *,
    only: Sequence[str] | None = None,
) -> dict:
    """
    Diff metadata dicts or aligned chunk lists; returns only changed keys.

    For chunk lists, keys are chunk indices (as int). For dict inputs, returns
    a flat {key: {before, after}} map.
    """
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        return _metadata_dict_diff(before, after, only=only)

    before_chunks = _normalize_chunks(before)  # type: ignore[arg-type]
    after_chunks = _normalize_chunks(after)  # type: ignore[arg-type]
    if len(before_chunks) != len(after_chunks):
        raise ValueError(
            f"chunk count mismatch: before={len(before_chunks)}, after={len(after_chunks)}"
        )

    per_chunk: dict[int, dict] = {}
    for i, (b, a) in enumerate(zip(before_chunks, after_chunks)):
        diff = _metadata_dict_diff(b.metadata or {}, a.metadata or {}, only=only)
        if diff:
            per_chunk[i] = diff
    return per_chunk


def _metadata_dict_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    only: Sequence[str] | None,
) -> dict:
    keys = set(before) | set(after)
    if only is not None:
        keys &= set(only)
    return {
        k: {"before": before.get(k), "after": after.get(k)}
        for k in sorted(keys)
        if before.get(k) != after.get(k)
    }


def _deterministic_vector(text: str, *, dim: int) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [float(b) / 255.0 for b in digest[:dim]]
    if len(raw) < dim:
        raw.extend([0.0] * (dim - len(raw)))
    return raw


class MockChunkerEmbeddingClient:
    """Deterministic sync client for SemanticChunker unit/integration tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [_deterministic_vector(t, dim=self.dim) for t in texts]


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder: identical text → identical vector (cosine-friendly)."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vectorize(self, text: str) -> List[float]:
        return _deterministic_vector(text, dim=self.dim)

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
def mock_chunker_embedding_client() -> MockChunkerEmbeddingClient:
    return MockChunkerEmbeddingClient(dim=8)


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
