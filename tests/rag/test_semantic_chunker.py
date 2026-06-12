"""Tests for rag.chunker.semantic_chunker."""

from __future__ import annotations

import pytest

from rag.chunker.semantic_chunker import SemanticChunker, _approx_token_len

pytestmark = pytest.mark.unit


def test_semantic_chunker_extracts_heading_path(sample_markdown: str):
    chunker = SemanticChunker(chunk_tokens=512, overlap_tokens=0, min_chunk_tokens=1)
    chunks = chunker.run(sample_markdown)

    assert len(chunks) >= 2
    headings = {(c.metadata or {}).get("heading_path") for c in chunks}
    assert any(h and "Alpha" in h for h in headings)
    assert any(h and "Beta" in h for h in headings)


def test_semantic_chunker_respects_token_limit():
    paragraphs = [f"topic{i} " * 25 for i in range(8)]
    text = "# Title\n\n" + "\n\n".join(paragraphs)
    chunker = SemanticChunker(chunk_tokens=80, overlap_tokens=0, min_chunk_tokens=1)
    chunks = chunker.run(text)

    assert len(chunks) >= 2
    token_lens = [_approx_token_len(c.content) for c in chunks]
    assert max(token_lens) <= 120
    assert sum(token_lens) > 80


def test_semantic_chunker_semantic_break_increases_chunk_count():
    text = """# Topic

Paragraph about machine learning models and neural networks training data.

Completely different subject about baking bread with yeast and flour."""

    tight = SemanticChunker(
        chunk_tokens=512,
        overlap_tokens=0,
        break_similarity=0.5,
        min_chunk_tokens=1,
    )
    loose = SemanticChunker(
        chunk_tokens=512,
        overlap_tokens=0,
        break_similarity=0.0,
        min_chunk_tokens=9999,
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
