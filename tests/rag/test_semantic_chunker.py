"""Tests for rag.chunker.semantic_chunker."""

from __future__ import annotations

from typing import List

import pytest

from rag.chunker.semantic_chunker import SemanticChunker, _approx_token_len
from tests.conftest import MockChunkerEmbeddingClient

pytestmark = pytest.mark.unit

_TOPIC_ML = [1.0, 0.0, 0.0, 0.0]
_TOPIC_BAKING = [0.0, 1.0, 0.0, 0.0]
_TOPIC_NEUTRAL = [0.5, 0.5, 0.0, 0.0]


class _TopicAwareEmbeddingClient:
    """Assigns orthogonal topic vectors so semantic-break tests are stable."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(w in lowered for w in ("machine", "neural", "learning", "training")):
                vectors.append(list(_TOPIC_ML))
            elif any(w in lowered for w in ("baking", "bread", "yeast", "flour")):
                vectors.append(list(_TOPIC_BAKING))
            else:
                vectors.append(list(_TOPIC_NEUTRAL))
        return vectors


@pytest.fixture
def topic_embedding_client() -> _TopicAwareEmbeddingClient:
    return _TopicAwareEmbeddingClient()


def test_semantic_chunker_extracts_heading_path(
    sample_markdown: str, mock_chunker_embedding_client: MockChunkerEmbeddingClient
):
    chunker = SemanticChunker(
        chunk_tokens=512,
        overlap_tokens=0,
        min_chunk_tokens=1,
        break_similarity=0.99,
        embedding_client=mock_chunker_embedding_client,
    )
    chunks = chunker.run(sample_markdown)

    assert len(chunks) >= 2
    headings = {(c.metadata or {}).get("heading_path") for c in chunks}
    assert any(h and "Alpha" in h for h in headings)
    assert any(h and "Beta" in h for h in headings)


def test_semantic_chunker_respects_token_limit(
    mock_chunker_embedding_client: MockChunkerEmbeddingClient,
):
    paragraphs = [f"topic{i} " * 25 for i in range(8)]
    text = "# Title\n\n" + "\n\n".join(paragraphs)
    chunker = SemanticChunker(
        chunk_tokens=80,
        overlap_tokens=0,
        min_chunk_tokens=1,
        embedding_client=mock_chunker_embedding_client,
    )
    chunks = chunker.run(text)

    assert len(chunks) >= 2
    token_lens = [_approx_token_len(c.content) for c in chunks]
    assert max(token_lens) <= 120
    assert sum(token_lens) > 80


def test_semantic_chunker_semantic_break_increases_chunk_count(
    topic_embedding_client: _TopicAwareEmbeddingClient,
):
    text = """# Topic

Paragraph about machine learning models and neural networks training data.

Completely different subject about baking bread with yeast and flour."""

    tight = SemanticChunker(
        chunk_tokens=512,
        overlap_tokens=0,
        break_similarity=0.5,
        min_chunk_tokens=1,
        embedding_client=topic_embedding_client,
    )
    loose = SemanticChunker(
        chunk_tokens=512,
        overlap_tokens=0,
        break_similarity=0.0,
        min_chunk_tokens=9999,
        embedding_client=topic_embedding_client,
    )

    tight_chunks = tight.run(text)
    loose_chunks = loose.run(text)
    assert len(tight_chunks) >= len(loose_chunks)

    reasons = [c.metadata.get("boundary_reason", "") for c in tight_chunks]
    assert any("semantic_break" in r for r in reasons)


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/rag/test_semantic_chunker.py -v
# ================================================================================================================
