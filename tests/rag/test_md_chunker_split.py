"""Tests for rag.chunker.split_paragraphs."""

from __future__ import annotations

import pytest

from rag.chunker.split_paragraphs import (
    ParagraphBoundaryMode,
    approx_token_len,
    detect_paragraph_boundary_mode,
    split_paragraphs,
)

pytestmark = pytest.mark.unit


def test_detect_single_line_mode_when_no_double_newlines() -> None:
    text = "# Title\nLine one.\nLine two."
    assert detect_paragraph_boundary_mode(text) is ParagraphBoundaryMode.SINGLE_LINE


def test_single_line_mode_splits_each_line() -> None:
    text = "# Fruits\nApple is good.\nBanana is yellow."
    paragraphs = split_paragraphs(text, max_paragraph_tokens=512)

    assert len(paragraphs) == 2
    assert paragraphs[0]["content"] == "Apple is good."
    assert paragraphs[1]["content"] == "Banana is yellow."


def test_blank_line_mode_keeps_single_newline_lines_together() -> None:
    text = "# Fruits\nApple is good.\nStill about apple.\n\nBanana is yellow."
    paragraphs = split_paragraphs(text, max_paragraph_tokens=512)

    assert len(paragraphs) == 2
    assert "Apple is good.\nStill about apple." in paragraphs[0]["content"]


def test_oversized_paragraph_falls_back_to_punctuation() -> None:
    sentence = "word " * 20
    text = f"# Title\n\n{sentence}. {sentence}. {sentence}."
    paragraphs = split_paragraphs(text, max_paragraph_tokens=30)

    assert len(paragraphs) >= 3
    assert all(approx_token_len(p["content"]) <= 30 for p in paragraphs)
