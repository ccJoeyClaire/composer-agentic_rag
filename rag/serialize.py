"""Structured JSON / JSONL serialization for indexed chunks and retrieval traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from .base import (
    TRACE_HYDE_DOCUMENT_KEY,
    TRACE_RERANKED_KEY,
    TRACE_RETRIEVED_KEY,
    TRACE_SMALL_RETRIEVED_KEY,
    TRACE_WORKING_QUERY_KEY,
    Chunk,
    RagResult,
)

META_RECORD_TYPE = "meta"
CHUNK_RECORD_TYPE = "chunk"
TRACE_RECORD_TYPE = "trace"


class ChunkRecord(TypedDict, total=False):
    """One indexed or retrieved chunk as a JSON-friendly record."""

    record_type: str
    rank: int
    chunk_id: str
    chunk_index: int
    chunk_role: str
    heading_path: str
    boundary_reason: str
    source: str
    content: str
    embed_text: str
    contextual_header: str
    predicted_questions: list[str]
    matched_chunk_ids: list[str]
    matched_small_content: str | list[str]
    score: float


class TraceStageRecord(TypedDict, total=False):
    """Retrieval pipeline stages for one query."""

    working_query: str
    hyde_document: str
    small_retrieved: list[ChunkRecord]
    retrieved: list[ChunkRecord]
    reranked: list[ChunkRecord]
    final: list[ChunkRecord]


@dataclass(frozen=True)
class IndexRunMeta:
    """Metadata written alongside an index dump."""

    profile_id: str
    collection: str
    config_path: str
    source: str


@dataclass(frozen=True)
class RetrieveRunMeta:
    """Metadata written alongside a retrieval trace dump."""

    profile_id: str
    collection: str
    config_path: str


class RetrieveTraceEntry(TypedDict):
    """One query's retrieval trace (stages only; run meta lives on the dump root)."""

    query: str
    top_k: int
    stages: TraceStageRecord


class RetrieveTraceDump(TypedDict):
    """Pretty-printed retrieval trace file (``indent=4`` JSON)."""

    meta: dict[str, object]
    traces: list[RetrieveTraceEntry]


class RetrieveTraceItem(TypedDict):
    """One query trace inside a pretty-printed JSON dump."""

    query: str
    top_k: int
    stages: TraceStageRecord


class RetrieveTraceDump(TypedDict):
    """Top-level shape for ``write_retrieve_traces_json`` output."""

    meta: dict[str, object]
    traces: list[RetrieveTraceItem]


def _meta_to_dict(meta: IndexRunMeta | RetrieveRunMeta) -> dict[str, object]:
    payload = asdict(meta)
    payload["record_type"] = META_RECORD_TYPE
    return payload


def chunk_to_record(
    chunk: Chunk,
    *,
    rank: int | None = None,
    include_vectors: bool = False,
) -> ChunkRecord:
    """Map a :class:`Chunk` to a JSON-serializable record."""
    meta = chunk.metadata or {}
    record: ChunkRecord = {
        "record_type": CHUNK_RECORD_TYPE,
        "content": chunk.content,
        "score": chunk.score,
    }
    if rank is not None:
        record["rank"] = rank

    for key in (
        "chunk_id",
        "chunk_index",
        "chunk_role",
        "heading_path",
        "boundary_reason",
        "source",
        "embed_text",
        "contextual_header",
        "predicted_questions",
        "matched_chunk_ids",
        "matched_small_content",
    ):
        if key in meta:
            record[key] = meta[key]  # type: ignore[literal-required]

    if include_vectors:
        vector = meta.get("vector") if isinstance(meta, dict) else None
        if vector is not None:
            record["vector"] = vector  # type: ignore[typeddict-unknown-key]

    return record


def chunks_to_records(
    chunks: list[Chunk],
    *,
    include_vectors: bool = False,
) -> list[ChunkRecord]:
    """Map many chunks to rank-preserving records."""
    return [
        chunk_to_record(chunk, rank=index + 1, include_vectors=include_vectors)
        for index, chunk in enumerate(chunks)
    ]


def trace_to_record(
    result: RagResult,
    *,
    top_k: int,
) -> RetrieveTraceEntry:
    """Map an :func:`RAGRetriever.aquery_trace` result to a structured record."""
    trace_meta = result.metadata or {}
    stages: TraceStageRecord = {}

    working_query = trace_meta.get(TRACE_WORKING_QUERY_KEY)
    if isinstance(working_query, str):
        stages["working_query"] = working_query

    hyde_document = trace_meta.get(TRACE_HYDE_DOCUMENT_KEY)
    if isinstance(hyde_document, str):
        stages["hyde_document"] = hyde_document

    small_retrieved = trace_meta.get(TRACE_SMALL_RETRIEVED_KEY)
    if isinstance(small_retrieved, list):
        stages["small_retrieved"] = chunks_to_records(small_retrieved)

    retrieved = trace_meta.get(TRACE_RETRIEVED_KEY)
    if isinstance(retrieved, list):
        stages["retrieved"] = chunks_to_records(retrieved)

    reranked = trace_meta.get(TRACE_RERANKED_KEY)
    if isinstance(reranked, list):
        stages["reranked"] = chunks_to_records(reranked)

    stages["final"] = chunks_to_records(result.chunks)

    return RetrieveTraceEntry(
        query=result.query,
        top_k=top_k,
        stages=stages,
    )


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, object], *, indent: int = 4) -> None:
    """Write a single pretty-printed JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def write_index_chunks_jsonl(
    path: Path,
    chunks: list[Chunk],
    *,
    meta: IndexRunMeta,
    include_vectors: bool = False,
) -> None:
    """Dump indexed chunks to JSONL with a leading meta line."""
    records: list[dict[str, object]] = [_meta_to_dict(meta)]
    records.extend(
        chunk_to_record(chunk, rank=index + 1, include_vectors=include_vectors)
        for index, chunk in enumerate(chunks)
    )
    write_jsonl(path, records)


def write_retrieve_traces_json(
    path: Path,
    results: list[RagResult],
    *,
    meta: RetrieveRunMeta,
    top_k: int,
) -> None:
    """Dump retrieval traces as pretty-printed JSON (``indent=4``)."""
    payload: RetrieveTraceDump = {
        "meta": _meta_to_dict(meta),
        "traces": [trace_to_record(result, top_k=top_k) for result in results],
    }
    write_json(path, payload)
