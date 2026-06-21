"""Tests for rag.serialize."""

from __future__ import annotations

import json

import pytest

from rag.base import (
    TRACE_RETRIEVED_KEY,
    TRACE_SMALL_RETRIEVED_KEY,
    Chunk,
    RagResult,
)
from rag.serialize import (
    META_RECORD_TYPE,
    IndexRunMeta,
    chunk_to_record,
    trace_to_record,
    write_index_chunks_jsonl,
)

pytestmark = pytest.mark.unit


def test_chunk_to_record_maps_metadata_and_rank() -> None:
    chunk = Chunk(
        content="body",
        metadata={
            "chunk_id": "doc.md::0",
            "heading_path": "Intro",
            "embed_text": "header\nbody",
        },
        score=0.75,
    )
    record = chunk_to_record(chunk, rank=1)
    assert record["rank"] == 1
    assert record["chunk_id"] == "doc.md::0"
    assert record["content"] == "body"
    assert record["embed_text"] == "header\nbody"
    assert record["score"] == 0.75


def test_trace_to_record_includes_small_retrieved_stage() -> None:
    small = Chunk(content="small hit", metadata={"chunk_id": "d::1"}, score=0.9)
    parent = Chunk(content="parent span", metadata={"chunk_id": "d::parent"}, score=0.9)
    result = RagResult(
        query="question",
        chunks=[parent],
        metadata={
            TRACE_SMALL_RETRIEVED_KEY: [small],
            TRACE_RETRIEVED_KEY: [parent],
        },
    )
    record = trace_to_record(
        result,
        profile_id="baseline",
        collection="getstart_codex_baseline",
        top_k=3,
    )
    stages = record["stages"]
    assert len(stages["small_retrieved"]) == 1
    assert stages["small_retrieved"][0]["chunk_id"] == "d::1"
    assert len(stages["final"]) == 1


def test_write_index_chunks_jsonl_leading_meta_line(tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    meta = IndexRunMeta(
        profile_id="baseline",
        collection="getstart_codex_baseline",
        config_path="/tmp/arg_config.yaml",
        source="demo.md",
    )
    write_index_chunks_jsonl(
        path,
        [Chunk(content="chunk", metadata={"chunk_id": "demo.md::0"})],
        meta=meta,
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["record_type"] == META_RECORD_TYPE
    assert first["profile_id"] == "baseline"
