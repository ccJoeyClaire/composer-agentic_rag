"""Tests for rag.document_augmentation.parent_builder (lazy Small-to-Big windows)."""

from __future__ import annotations

import pytest

from rag.base import Chunk
from rag.document_augmentation.parent_builder import (
    ANCHOR_WINDOW_KEY,
    CHUNK_ID_KEY,
    CHUNK_INDEX_KEY,
    CHUNK_ROLE_KEY,
    assign_parent_chunks,
    cluster_overlapping_hits,
    materialize_parent_content,
    merge_windows,
    trim_to_token_budget,
    windows_overlap,
)
from tests.conftest import make_chunk, make_small_chunks

pytestmark = pytest.mark.unit


def test_assign_parent_chunks_empty():
    assert assign_parent_chunks([]) == []


def test_assign_parent_chunks_stable_ids_and_windows():
    chunks = make_small_chunks(["aaa", "bbb", "ccc"], source="kb.md")
    result = assign_parent_chunks(chunks, parent_token_budget=512)

    assert len(result) == 3
    assert result[0].metadata[CHUNK_ID_KEY] == "kb.md::0"
    assert result[1].metadata[CHUNK_ID_KEY] == "kb.md::1"
    assert result[2].metadata[CHUNK_ROLE_KEY] == "small"

    window = result[1].metadata[ANCHOR_WINDOW_KEY]
    assert window["anchor_id"] == "kb.md::1"
    assert "kb.md::1" in window["member_ids"]


def test_assign_parent_chunks_respects_section_boundaries():
    chunks = [
        make_chunk("a", metadata={"source": "x.md", "heading_path": "H1"}),
        make_chunk("b", metadata={"source": "x.md", "heading_path": "H2"}),
    ]
    result = assign_parent_chunks(chunks, parent_token_budget=9999)

    w0 = result[0].metadata[ANCHOR_WINDOW_KEY]["member_ids"]
    w1 = result[1].metadata[ANCHOR_WINDOW_KEY]["member_ids"]
    assert w0 == ["x.md::0"]
    assert w1 == ["x.md::1"]


def test_windows_overlap_and_merge():
    a = {"anchor_id": "doc::1", "member_ids": ["doc::1", "doc::2"]}
    b = {"anchor_id": "doc::2", "member_ids": ["doc::2", "doc::3"]}
    assert windows_overlap(a, b) is True

    merged = merge_windows([a, b])
    assert merged["member_ids"] == ["doc::1", "doc::2", "doc::3"]
    assert merged["anchor_id"].startswith("merged:")


def test_cluster_overlapping_hits_transitive_closure():
    hit_a = make_chunk(
        "a",
        metadata={
            CHUNK_ID_KEY: "d::0",
            ANCHOR_WINDOW_KEY: {"anchor_id": "d::0", "member_ids": ["d::0", "d::1"]},
        },
        score=0.9,
    )
    hit_b = make_chunk(
        "b",
        metadata={
            CHUNK_ID_KEY: "d::1",
            ANCHOR_WINDOW_KEY: {"anchor_id": "d::1", "member_ids": ["d::1", "d::2"]},
        },
        score=0.8,
    )
    hit_c = make_chunk(
        "c",
        metadata={
            CHUNK_ID_KEY: "d::5",
            ANCHOR_WINDOW_KEY: {"anchor_id": "d::5", "member_ids": ["d::5"]},
        },
        score=0.7,
    )

    clusters = cluster_overlapping_hits([hit_a, hit_b, hit_c])
    assert len(clusters) == 2
    assert {len(c) for c in clusters} == {1, 2}


def test_materialize_parent_content_strips_char_overlap():
    members = [
        Chunk(
            content="hello world",
            metadata={CHUNK_INDEX_KEY: 0, "start": 0, "end": 11},
        ),
        Chunk(
            content="world again",
            metadata={CHUNK_INDEX_KEY: 1, "start": 6, "end": 17},
        ),
    ]
    text = materialize_parent_content(members)
    assert "hello world again" in text.replace("\n\n", " ")
    assert text.count("world") == 1


def test_trim_to_token_budget():
    long_text = "word " * 500
    trimmed = trim_to_token_budget(long_text, budget=10)
    assert len(trimmed) < len(long_text)
    assert trim_to_token_budget("", budget=10) == ""
