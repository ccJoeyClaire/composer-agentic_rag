"""Tests for rag.document_augmentation.context_enricher."""

from __future__ import annotations

import pytest

from rag.document_augmentation.context_enricher import (
    ContextualEnricher,
    build_contextual_header,
    build_embed_text,
)
from tests.conftest import make_chunk

pytestmark = pytest.mark.unit


def test_build_contextual_header_includes_doc_and_section():
    meta = {
        "doc_title": "Handbook",
        "source": "handbook.md",
        "heading_path": "Intro > Setup",
        "doc_keywords": ["Intro", "Setup"],
    }
    header = build_contextual_header(meta, source="handbook.md")
    assert "Document: Handbook" in header
    assert "Section: Intro > Setup" in header
    assert "Keywords:" in header


def test_build_embed_text_joins_header_and_body():
    body = "Main content here."
    embed = build_embed_text("Document: demo", body)
    assert embed.startswith("Document: demo")
    assert body in embed


@pytest.mark.asyncio
async def test_aenrich_for_index_sets_embed_text():
    enricher = ContextualEnricher()
    chunks = [
        make_chunk("body", metadata={"heading_path": "Chapter > Part"}),
    ]
    out = await enricher.aenrich_for_index(chunks, source="guide.md")

    assert out[0].metadata["doc_title"] == "guide"
    assert "embed_text" in out[0].metadata
    assert "body" in out[0].metadata["embed_text"]
    assert "contextual_header" in out[0].metadata


@pytest.mark.asyncio
async def test_aenrich_chunks_prepends_header_on_retrieve():
    enricher = ContextualEnricher(prepend_on_retrieve=True)
    chunk = make_chunk(
        "answer text",
        metadata={
            "contextual_header": "Document: demo\nSection: FAQ",
        },
    )
    out = await enricher.aenrich_chunks([chunk])
    assert out[0].content.startswith("Document: demo")
    assert "answer text" in out[0].content


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/rag/test_contextual_enricher.py -v
# ================================================================================================================
