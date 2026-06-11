"""Tests for rag.document_augmentation.parent_builder (lazy Small-to-Big windows)."""

from __future__ import annotations

import json
import os

import pytest

from rag.base import Chunk
from rag.document_augmentation.parent_builder import (
    ANCHOR_WINDOW_KEY,
    CHUNK_ID_KEY,
    CHUNK_INDEX_KEY,
    CHUNK_ROLE_KEY,
    SECTION_ID_KEY,
    assign_parent_chunks,
    cluster_overlapping_hits,
    materialize_parent_content,
    merge_windows,
    windows_overlap,
)
from tests.conftest import make_chunk, make_small_chunks, metadata_diff, show_metadata

pytestmark = pytest.mark.unit


def _assert_indexed_chunk_metadata(
    chunk: Chunk,
    *,
    index: int,
    source: str,
    heading_path: str | None,
    expect_anchor_window: bool = True,
) -> None:
    """Assert the metadata contract written by assign_parent_chunks."""
    meta = chunk.metadata
    assert meta[CHUNK_ID_KEY] == f"{source}::{index}"
    assert meta[CHUNK_INDEX_KEY] == index
    expected_section = (
        f"{source}::{heading_path}" if heading_path else f"{source}::__root__"
    )
    assert meta[SECTION_ID_KEY] == expected_section
    assert meta[CHUNK_ROLE_KEY] == "small"
    assert meta.get("source") == source

    if not expect_anchor_window:
        assert ANCHOR_WINDOW_KEY not in meta
        return

    window = meta[ANCHOR_WINDOW_KEY]
    assert set(window) == {"anchor_id", "member_ids"}
    assert window["anchor_id"] == meta[CHUNK_ID_KEY]
    member_ids = window["member_ids"]
    assert isinstance(member_ids, list)
    assert member_ids
    assert meta[CHUNK_ID_KEY] in member_ids
    member_indices = [int(mid.rsplit("::", 1)[-1]) for mid in member_ids]
    assert member_indices == sorted(member_indices)


def test_assign_parent_chunks_empty():
    assert assign_parent_chunks([]) == []


def test_assign_parent_chunks_metadata_contract():
    chunks = make_small_chunks(["aaa", "bbb", "ccc"], source="kb.md")
    result = assign_parent_chunks(chunks, parent_token_budget=512)

    assert len(result) == 3
    for i, chunk in enumerate(result):
        _assert_indexed_chunk_metadata(
            chunk,
            index=i,
            source="kb.md",
            heading_path="Section A",
        )

    window = result[1].metadata[ANCHOR_WINDOW_KEY]
    assert window["anchor_id"] == "kb.md::1"
    assert "kb.md::2" in window["member_ids"]


def test_assign_parent_chunks_skips_windows_when_budget_disabled():
    chunks = make_small_chunks(["a", "b"], source="kb.md")
    result = assign_parent_chunks(chunks, parent_token_budget=0)

    for i, chunk in enumerate(result):
        _assert_indexed_chunk_metadata(
            chunk,
            index=i,
            source="kb.md",
            heading_path="Section A",
            expect_anchor_window=False,
        )


def test_assign_parent_chunks_respects_section_boundaries():
    chunks = [
        make_chunk("a", metadata={"source": "x.md", "heading_path": "H1"}),
        make_chunk("b", metadata={"source": "x.md", "heading_path": "H2"}),
    ]
    result = assign_parent_chunks(chunks, parent_token_budget=9999)

    _assert_indexed_chunk_metadata(
        result[0], index=0, source="x.md", heading_path="H1"
    )
    _assert_indexed_chunk_metadata(
        result[1], index=1, source="x.md", heading_path="H2"
    )

    w0 = result[0].metadata[ANCHOR_WINDOW_KEY]["member_ids"]
    w1 = result[1].metadata[ANCHOR_WINDOW_KEY]["member_ids"]
    assert w0 == ["x.md::0"]
    assert w1 == ["x.md::1"]


def test_assign_parent_chunks_token_budget_limits_window():
    # Each chunk is 2 tokens; budget 3 fits only the anchor, not neighbors.
    chunks = make_small_chunks(["one two", "three four", "five six"], source="s.md")
    result = assign_parent_chunks(chunks, parent_token_budget=3)

    center_window = result[1].metadata[ANCHOR_WINDOW_KEY]["member_ids"]
    assert center_window == ["s.md::1"]


def test_windows_overlap_and_merge():
    a = {"anchor_id": "doc::1", "member_ids": ["doc::1", "doc::2"]}
    b = {"anchor_id": "doc::2", "member_ids": ["doc::2", "doc::3"]}
    assert windows_overlap(a, b) is True

    merged = merge_windows([a, b])
    assert merged["member_ids"] == ["doc::1", "doc::2", "doc::3"]
    assert merged["anchor_id"] == "doc::1"  # 纯合并，不加 merged: 前缀


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
    assert {len(c.hits) for c in clusters} == {1, 2}
    merged = next(c for c in clusters if len(c.hits) == 2).merged_window()
    assert merged is not None
    assert merged["member_ids"] == ["d::0", "d::1", "d::2"]


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


@pytest.mark.debug_probe
@pytest.mark.skipif(
    os.environ.get("DEBUG_PROBES") != "1",
    reason="Set DEBUG_PROBES=1 to run (see test docstring)",
)
def test_debug_assign_parent_chunks_output():
    """
    Inspect assign_parent_chunks metadata.

    PowerShell:
        $env:DEBUG_PROBES=1; python -m pytest tests/rag/test_parent_builder.py::test_debug_assign_parent_chunks_output -s

    cmd:
        set DEBUG_PROBES=1 && python -m pytest tests/rag/test_parent_builder.py::test_debug_assign_parent_chunks_output -s
    """
    chunks = make_small_chunks(["aaa", "bbb", "ccc"], source="kb.md")
    before = [make_chunk(c.content, metadata=dict(c.metadata or {})) for c in chunks]
    after = assign_parent_chunks(chunks, parent_token_budget=512)

    show_metadata(after, only=[CHUNK_ID_KEY, ANCHOR_WINDOW_KEY])
    diff = metadata_diff(before, after)
    print("\n--- metadata_diff ---")
    print(json.dumps(diff, indent=2, sort_keys=True, ensure_ascii=False))

#================================================================================================================
# 启用探针：
# PowerShell:
# $env:DEBUG_PROBES=1; python -m pytest tests/rag/test_parent_builder.py::test_debug_assign_parent_chunks_output -s
# cmd:
# set DEBUG_PROBES=1 && python -m pytest tests/rag/test_parent_builder.py::test_debug_assign_parent_chunks_output -s
#================================================================================================================