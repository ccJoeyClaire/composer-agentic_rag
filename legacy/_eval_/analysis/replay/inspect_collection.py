"""Inspect an indexed Qdrant collection after eval (chunk stats + payload preview).

Smoke workflow (after ``python -m _eval_.rag_eval.run``):

1. ``python -m _eval_.analysis.replay.inspect_collection --collection pooleval_nfcorpus_token``
2. Compare ``token`` vs ``semantic`` — ``chunks_per_doc`` should differ when paragraph
   structure is present.

Library usage::

    from _eval_.analysis.replay.inspect_collection import (
        InspectCollectionRequest,
        inspect_collection,
    )

    result = await inspect_collection(
        InspectCollectionRequest(
            collection="pooleval_nfcorpus_token",
            summary_only=True,
        )
    )
    print(result["summary"]["chunks_per_doc"])

Run (offline fixture demo, no Qdrant)::

    python -m _eval_.analysis.replay.inspect_collection

CLI (live Qdrant, pass any flag)::

    python -m _eval_.analysis.replay.inspect_collection \\
      --collection pooleval_nfcorpus_semantic \\
      --limit 20 \\
      [--doc-id DOC_ID] [--source TITLE] [--json] [--summary-only]
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TypedDict

from rag.base import BaseVectorStore, Chunk
from rag.document_augmentation.parent_builder import CHUNK_ROLE_KEY

from _eval_.data_preparing.beir import DOC_ID_META_KEY, SOURCE_META_KEY, resolve_chunk_doc_id
from _eval_.analysis.replay.rag_factory import open_qdrant_store

_SUMMARY_SCROLL_CAP = 2000


class IndexSummary(TypedDict):
    """Aggregate stats over a page (or scroll batch) of indexed chunks."""

    total_listed: int
    chunks_per_doc: dict[str, int]
    heading_path_nonempty_ratio: float
    chunk_role_counts: dict[str, int]
    avg_content_len: float
    avg_embed_text_len: float


class InspectChunkRecord(TypedDict):
    content: str
    metadata: dict[str, object]
    score: float


class InspectCollectionResult(TypedDict):
    """Structured outcome of :func:`inspect_collection`."""

    collection: str
    summary: IndexSummary
    chunks: list[InspectChunkRecord]
    next_offset: str | None


@dataclass(frozen=True)
class InspectCollectionRequest:
    """Inputs for :func:`inspect_collection` — explicit, importable, no argparse."""

    collection: str
    limit: int = 20
    doc_id: str | None = None
    source: str | None = None
    summary_only: bool = False
    summary_scroll_cap: int = _SUMMARY_SCROLL_CAP


def summarize_chunks(chunks: list[Chunk]) -> IndexSummary:
    """Compute index-shape stats for one chunk page."""
    if not chunks:
        return IndexSummary(
            total_listed=0,
            chunks_per_doc={},
            heading_path_nonempty_ratio=0.0,
            chunk_role_counts={},
            avg_content_len=0.0,
            avg_embed_text_len=0.0,
        )

    chunks_per_doc: Counter[str] = Counter()
    chunk_role_counts: Counter[str] = Counter()
    heading_nonempty = 0
    content_lens: list[int] = []
    embed_lens: list[int] = []

    for chunk in chunks:
        meta = chunk.metadata or {}
        doc_id = resolve_chunk_doc_id(meta) or "(missing)"
        chunks_per_doc[doc_id] += 1

        heading = str(meta.get("heading_path") or "").strip()
        if heading:
            heading_nonempty += 1

        role = meta.get(CHUNK_ROLE_KEY)
        if role:
            chunk_role_counts[str(role)] += 1

        content_lens.append(len(chunk.content))
        embed_text = str(meta.get("embed_text") or chunk.content)
        embed_lens.append(len(embed_text))

    total = len(chunks)
    return IndexSummary(
        total_listed=total,
        chunks_per_doc=dict(chunks_per_doc),
        heading_path_nonempty_ratio=heading_nonempty / total,
        chunk_role_counts=dict(chunk_role_counts),
        avg_content_len=sum(content_lens) / total,
        avg_embed_text_len=sum(embed_lens) / total,
    )


def chunk_to_record(chunk: Chunk) -> InspectChunkRecord:
    return InspectChunkRecord(
        content=chunk.content,
        metadata=dict(chunk.metadata or {}),
        score=chunk.score,
    )


async def scroll_collection_chunks(
    store: BaseVectorStore,
    *,
    cap: int,
    doc_id: str | None = None,
    source: str | None = None,
) -> list[Chunk]:
    """Scroll up to ``cap`` chunks with optional metadata filters."""
    collected: list[Chunk] = []
    offset: str | None = None
    while len(collected) < cap:
        page_size = min(256, cap - len(collected))
        page, offset = await store.alist_chunks(
            limit=page_size,
            offset=offset,
            doc_id=doc_id,
            source=source,
        )
        if not page:
            break
        collected.extend(page)
        if offset is None:
            break
    return collected[:cap]


async def inspect_collection(
    request: InspectCollectionRequest,
    *,
    store: BaseVectorStore | None = None,
) -> InspectCollectionResult:
    """Load chunks from a collection and return summary stats plus a chunk page.

    Opens (and closes) a Qdrant client when ``store`` is omitted. Pass an
    existing store to reuse a connection or use a test fake.

    Args:
        request: Collection name, filters, and listing mode.
        store: Optional vector store; defaults to :func:`open_qdrant_store`.

    Returns:
        Summary over the loaded chunk batch, serialized chunk records, and
        pagination offset (``None`` when ``summary_only`` or no further page).
    """
    owns_store = store is None
    active_store = store or open_qdrant_store(request.collection)
    try:
        if request.summary_only:
            chunks = await scroll_collection_chunks(
                active_store,
                cap=request.summary_scroll_cap,
                doc_id=request.doc_id,
                source=request.source,
            )
            return InspectCollectionResult(
                collection=request.collection,
                summary=summarize_chunks(chunks),
                chunks=[chunk_to_record(chunk) for chunk in chunks],
                next_offset=None,
            )

        chunks, next_offset = await active_store.alist_chunks(
            limit=request.limit,
            doc_id=request.doc_id,
            source=request.source,
        )
        return InspectCollectionResult(
            collection=request.collection,
            summary=summarize_chunks(chunks),
            chunks=[chunk_to_record(chunk) for chunk in chunks],
            next_offset=next_offset,
        )
    finally:
        if owns_store:
            await active_store.aclose()


def format_summary_text(summary: IndexSummary) -> str:
    lines = [
        f"total_listed: {summary['total_listed']}",
        f"heading_path_nonempty_ratio: {summary['heading_path_nonempty_ratio']:.3f}",
        f"avg_content_len: {summary['avg_content_len']:.1f}",
        f"avg_embed_text_len: {summary['avg_embed_text_len']:.1f}",
    ]
    if summary["chunk_role_counts"]:
        lines.append(f"chunk_role_counts: {summary['chunk_role_counts']}")
    lines.append("chunks_per_doc (top 10):")
    for doc_id, count in sorted(
        summary["chunks_per_doc"].items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]:
        lines.append(f"  {doc_id}: {count}")
    return "\n".join(lines)


def _format_chunk_row(rank: int, record: InspectChunkRecord, *, preview_len: int = 80) -> str:
    meta = record["metadata"]
    doc_id = resolve_chunk_doc_id(meta) or "-"
    role = str(meta.get(CHUNK_ROLE_KEY) or "-")
    source = str(meta.get(SOURCE_META_KEY) or "-")
    preview = record["content"].replace("\n", " ")[:preview_len]
    return f"{rank:>4} | {doc_id:<12} | {role:<8} | {source[:24]:<24} | {preview}"


def format_inspect_result(
    result: InspectCollectionResult,
    *,
    summary_only: bool = False,
) -> str:
    """Human-readable table for CLI or notebooks."""
    parts = [format_summary_text(result["summary"])]
    if summary_only:
        return "\n".join(parts)

    parts.append("")
    parts.append(f"{'rank':>4} | {'doc_id':<12} | {'role':<8} | {'source':<24} | preview")
    parts.append("-" * 80)
    for rank, record in enumerate(result["chunks"], start=1):
        parts.append(_format_chunk_row(rank, record))
    if result["next_offset"]:
        parts.append(f"\n(next page offset: {result['next_offset']})")
    return "\n".join(parts)


if __name__ == "__main__":
    import argparse
    import asyncio
    import json
    import sys

    def _inspect_result_to_json_dict(
        result: InspectCollectionResult,
        *,
        summary_only: bool = False,
    ) -> dict[str, object]:
        if summary_only:
            return dict(result["summary"])
        return {
            "collection": result["collection"],
            "summary": result["summary"],
            "next_offset": result["next_offset"],
            "chunks": result["chunks"],
        }

    def _request_from_cli(args: argparse.Namespace) -> InspectCollectionRequest:
        return InspectCollectionRequest(
            collection=args.collection,
            limit=args.limit,
            doc_id=args.doc_id,
            source=args.source,
            summary_only=args.summary_only,
        )

    async def _run_cli(
        request: InspectCollectionRequest,
        *,
        as_json: bool = False,
    ) -> int:
        result = await inspect_collection(request)
        if as_json:
            print(
                json.dumps(
                    _inspect_result_to_json_dict(
                        result, summary_only=request.summary_only
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(format_inspect_result(result, summary_only=request.summary_only))
        return 0

    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Inspect chunks stored in a post-eval Qdrant collection.",
        )
        parser.add_argument("--collection", required=True, help="Qdrant collection name")
        parser.add_argument("--limit", type=int, default=20, help="Page size for chunk listing")
        parser.add_argument("--doc-id", help=f"Filter on metadata.{DOC_ID_META_KEY}")
        parser.add_argument("--source", help=f"Filter on metadata.{SOURCE_META_KEY}")
        parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
        parser.add_argument(
            "--summary-only",
            action="store_true",
            help=f"Scroll up to {_SUMMARY_SCROLL_CAP} points and print stats only",
        )
        return parser

    def main(argv: list[str] | None = None) -> int:
        args = _build_parser().parse_args(argv)
        request = _request_from_cli(args)
        return asyncio.run(_run_cli(request, as_json=args.json))

    if len(sys.argv) > 1:
        raise SystemExit(main())

    def _demo_fixture_chunks() -> list[Chunk]:
        """Synthetic indexed chunks for offline demo (no Qdrant)."""
        return [
            Chunk(
                "Vitamin D supplementation reduces fracture risk in elderly patients.",
                metadata={
                    "doc_id": "MED-10",
                    "source": "Vitamin D and bone health",
                    "heading_path": "Introduction",
                    CHUNK_ROLE_KEY: "small",
                    "embed_text": "Vitamin D supplementation reduces fracture risk.",
                },
                score=0.0,
            ),
            Chunk(
                "Calcium intake alone does not prevent osteoporosis without vitamin D.",
                metadata={
                    "doc_id": "MED-10",
                    "source": "Vitamin D and bone health",
                    "heading_path": "Mechanism",
                    CHUNK_ROLE_KEY: "parent",
                },
                score=0.0,
            ),
            Chunk(
                "Unrelated passage about seasonal allergies and antihistamines.",
                metadata={
                    "doc_id": "MED-42",
                    "source": "Allergy management review",
                    "heading_path": "",
                    CHUNK_ROLE_KEY: "small",
                },
                score=0.0,
            ),
        ]

    async def _run_offline_demo() -> None:
        from tests.fakes.vector_store import InMemoryVectorStore

        store = InMemoryVectorStore()
        demo_chunks = _demo_fixture_chunks()
        await store.aadd_chunks(demo_chunks, [[1.0] for _ in demo_chunks])

        summary_result = await inspect_collection(
            InspectCollectionRequest(
                collection="demo_nfcorpus_token",
                summary_only=True,
                summary_scroll_cap=100,
            ),
            store=store,
        )
        print("=== Usage 1: InspectCollectionRequest(summary_only=True) ===\n")
        print(format_inspect_result(summary_result, summary_only=True))

        page_result = await inspect_collection(
            InspectCollectionRequest(
                collection="demo_nfcorpus_token",
                limit=5,
                doc_id="MED-10",
            ),
            store=store,
        )
        print("\n=== Usage 2: InspectCollectionRequest(limit=5, doc_id='MED-10') ===\n")
        print(format_inspect_result(page_result))

    asyncio.run(_run_offline_demo())
    print(
        "\nLive Qdrant: python -m _eval_.analysis.replay.inspect_collection "
        "--collection pooleval_nfcorpus_token --summary-only"
    )
