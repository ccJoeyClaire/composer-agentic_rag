"""Tests for rag.document_augmentation.predict_question."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.document_augmentation.predict_question import (
    PredictQuestionEnricher,
    append_questions_to_embed_text,
    augment_chunks_with_questions,
    parse_predicted_questions,
    predict_questions_for_chunk,
)
from tests.conftest import make_chunk

pytestmark = pytest.mark.unit


def test_parse_predicted_questions_from_json():
    raw = '{"questions": ["What is RAG?", "How does chunking work?", "What is RAG?"]}'
    assert parse_predicted_questions(raw) == [
        "What is RAG?",
        "How does chunking work?",
    ]


def test_parse_predicted_questions_invalid_json():
    assert parse_predicted_questions("not json") == []


def test_append_questions_to_embed_text():
    out = append_questions_to_embed_text(
        "Document: demo\n\nbody",
        ["Q1?", "Q2?"],
    )
    assert "Document: demo" in out
    assert "Possible questions:" in out
    assert "Q1?" in out


@pytest.mark.asyncio
async def test_predict_questions_for_chunk_calls_llm():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"questions": ["How to install?"]}'
    mock_llm.arequest_llm = AsyncMock(return_value=mock_response)

    chunk = make_chunk("Install with pip.", metadata={"heading_path": "Setup"})
    questions = await predict_questions_for_chunk(chunk, mock_llm, num_questions=2)

    assert questions == ["How to install?"]
    mock_llm.arequest_llm.assert_awaited_once()
    call_kwargs = mock_llm.arequest_llm.await_args.kwargs
    assert call_kwargs["json_output"] is True


@pytest.mark.asyncio
async def test_augment_chunks_runs_llm_calls_concurrently():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"questions": ["Q?"]}'
    in_flight = 0
    peak = 0

    async def _slow_request(*_args, **_kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return mock_response

    mock_llm.arequest_llm = AsyncMock(side_effect=_slow_request)

    chunks = [make_chunk(f"chunk {i}.") for i in range(4)]
    out = await augment_chunks_with_questions(
        chunks,
        mock_llm,
        max_concurrency=499,
    )

    assert len(out) == 4
    assert peak > 1
    assert mock_llm.arequest_llm.await_count == 4


def test_bounded_concurrency_caps_at_499():
    from rag.document_augmentation.predict_question import _bounded_concurrency

    assert _bounded_concurrency(500) == 499
    assert _bounded_concurrency(0) == 1


@pytest.mark.asyncio
async def test_augment_chunks_with_questions_sets_metadata_and_embed_text():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"questions": ["What is alpha?"]}'
    mock_llm.arequest_llm = AsyncMock(return_value=mock_response)

    chunks = [make_chunk("Alpha content.", metadata={"embed_text": "header\n\nAlpha content."})]
    out = await augment_chunks_with_questions(chunks, mock_llm)

    assert out[0].metadata["predicted_questions"] == ["What is alpha?"]
    assert "Possible questions:" in out[0].metadata["embed_text"]
    assert "What is alpha?" in out[0].metadata["embed_text"]


@pytest.mark.asyncio
async def test_predict_question_enricher_aenrich_for_index():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"questions": ["Why beta?"]}'
    mock_llm.arequest_llm = AsyncMock(return_value=mock_response)

    enricher = PredictQuestionEnricher(llm_client=mock_llm)
    out = await enricher.aenrich_for_index(
        [make_chunk("Beta details.")],
        source="doc.md",
    )

    assert out[0].metadata["source"] == "doc.md"
    assert out[0].metadata["predicted_questions"] == ["Why beta?"]


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/rag/test_predict_question.py -v
# ================================================================================================================
