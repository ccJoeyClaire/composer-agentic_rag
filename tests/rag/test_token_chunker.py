"""Tests for rag.chunker.token_chunker."""

from __future__ import annotations

import pytest

from rag.chunker.token_chunker import TokenChunker, _token_len

pytestmark = pytest.mark.unit


def test_token_chunker_respects_token_limit() -> None:
    text = "word " * 500
    chunker = TokenChunker(chunk_tokens=50, overlap_tokens=0)
    chunks = chunker.run(text)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert _token_len(chunk.content) <= 50


def test_token_chunker_overlap_increases_chunk_count() -> None:
    text = "x " * 400
    no_overlap = TokenChunker(chunk_tokens=40, overlap_tokens=0).run(text)
    with_overlap = TokenChunker(chunk_tokens=40, overlap_tokens=10).run(text)

    assert len(with_overlap) > len(no_overlap)


def test_token_chunker_ignores_markdown_structure() -> None:
    text = "# Title\n\nBody paragraph one.\n\nBody paragraph two."
    chunker = TokenChunker(chunk_tokens=512, overlap_tokens=0)
    chunks = chunker.run(text)

    assert len(chunks) == 1
    assert chunks[0].metadata.get("heading_path") is None
    assert "# Title" in chunks[0].content
