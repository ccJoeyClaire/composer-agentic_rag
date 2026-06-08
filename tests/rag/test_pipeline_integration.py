"""Integration tests for RAGIndexer / RAGRetriever (mock embedder + in-memory Qdrant)."""

from __future__ import annotations

import pytest

from rag.chunker.semantic_chunker import SemanticChunker
from rag.core import RAGIndexer, RAGRetriever
from rag.document_augmentation.context_enricher import ContextualEnricher
from rag.retriever.vector_retriever import VectorRetriever

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_indexer_stores_chunks(mock_embedder, in_memory_qdrant_store):
    indexer = RAGIndexer(
        chunker=SemanticChunker(chunk_tokens=128, overlap_tokens=0, min_chunk_tokens=1),
        embedder=mock_embedder,
        store=in_memory_qdrant_store,
        contextual_enricher=ContextualEnricher(),
        small_to_big_parent_tokens=256,
    )
    text = "# Doc\n\nRAG pipeline indexes markdown for retrieval."
    ok = await indexer.aindex(text, source="doc.md")

    assert ok
    assert not await indexer.averify_index("missing.md")


@pytest.mark.asyncio
async def test_retriever_returns_matching_chunk(mock_embedder, in_memory_qdrant_store):
    indexer = RAGIndexer(
        chunker=SemanticChunker(chunk_tokens=256, overlap_tokens=0, min_chunk_tokens=1),
        embedder=mock_embedder,
        store=in_memory_qdrant_store,
    )
    needle = "unique phrase about observability dashboards"
    await indexer.aindex(f"# Note\n\n{needle}", source="obs.md")

    retriever = RAGRetriever(
        retriever=VectorRetriever(
            embedder=mock_embedder,
            store=in_memory_qdrant_store,
        ),
        recall_n=5,
    )
    hits = await retriever.aquery(needle, top_k=1)

    assert len(hits) == 1
    assert needle in hits[0].content


@pytest.mark.asyncio
async def test_retriever_applies_top_k_without_reranker(mock_embedder, in_memory_qdrant_store):
    indexer = RAGIndexer(
        chunker=SemanticChunker(chunk_tokens=64, overlap_tokens=0, min_chunk_tokens=1),
        embedder=mock_embedder,
        store=in_memory_qdrant_store,
    )
    text = "\n\n".join(f"Paragraph {i} about topic {i}." for i in range(6))
    await indexer.aindex(f"# Long\n\n{text}", source="long.md")

    retriever = RAGRetriever(
        retriever=VectorRetriever(
            embedder=mock_embedder,
            store=in_memory_qdrant_store,
        ),
    )
    hits = await retriever.aquery("Paragraph 3 about topic 3.", top_k=2)
    assert len(hits) <= 2
