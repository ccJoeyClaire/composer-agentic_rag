"""Pure helpers for replay query output: stage tables and gold-doc rank lookup."""

from __future__ import annotations

from typing import Literal, TypedDict

from rag.base import Chunk
from rag.document_augmentation.parent_builder import CHUNK_ROLE_KEY

from _eval_.data_preparing.beir import DocId, resolve_chunk_doc_id
from _eval_.rag_eval.pipeline import ranked_doc_ids

StageName = Literal["retrieved", "reranked", "final"]


class GoldRankRow(TypedDict):
    """Per-gold-doc rank at each pipeline stage."""

    doc_id: DocId
    rank_retrieved: int | None
    rank_reranked: int | None
    rank_final: int | None
    in_top_k: bool


def rank_of_doc(chunks: list[Chunk], doc_id: DocId) -> int | None:
    """Return 1-based deduplicated doc rank, or ``None`` if absent."""
    ranked = ranked_doc_ids(chunks)
    try:
        return ranked.index(doc_id) + 1
    except ValueError:
        return None


def gold_rank_report(
    gold_doc_ids: set[DocId],
    *,
    retrieved: list[Chunk],
    reranked: list[Chunk] | None,
    final: list[Chunk],
    top_k: int,
) -> list[GoldRankRow]:
    """Map each gold doc id to its rank at retrieved / reranked / final stages."""
    rows: list[GoldRankRow] = []
    for doc_id in sorted(gold_doc_ids):
        rank_retrieved = rank_of_doc(retrieved, doc_id)
        rank_reranked = rank_of_doc(reranked, doc_id) if reranked is not None else None
        rank_final = rank_of_doc(final, doc_id)
        best_rank = rank_final if rank_final is not None else rank_retrieved
        rows.append(
            GoldRankRow(
                doc_id=doc_id,
                rank_retrieved=rank_retrieved,
                rank_reranked=rank_reranked,
                rank_final=rank_final,
                in_top_k=best_rank is not None and best_rank <= top_k,
            )
        )
    return rows


def format_stage_table(
    chunks: list[Chunk],
    *,
    stage: StageName,
    preview_len: int = 80,
) -> str:
    """Format one pipeline stage as a fixed-width rank table."""
    lines = [
        f"[{stage}]",
        f"{'rank':>4} | {'doc_id':<12} | {'chunk_role':<8} | {'score':>8} | content_preview({preview_len})",
        "-" * 80,
    ]
    for rank, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        doc_id = resolve_chunk_doc_id(meta) or "-"
        role = str(meta.get(CHUNK_ROLE_KEY) or "-")
        preview = chunk.content.replace("\n", " ")[:preview_len]
        lines.append(
            f"{rank:>4} | {doc_id:<12} | {role:<8} | {chunk.score:8.4f} | {preview}"
        )
    return "\n".join(lines)


def format_gold_table(rows: list[GoldRankRow]) -> str:
    """Format gold-doc rank comparison across stages."""
    lines = [
        "gold doc_id | rank@retrieved | rank@reranked | rank@final | in_top_k",
        "-" * 72,
    ]
    for row in rows:
        lines.append(
            f"{row['doc_id']:<12} | "
            f"{_fmt_rank(row['rank_retrieved']):>14} | "
            f"{_fmt_rank(row['rank_reranked']):>13} | "
            f"{_fmt_rank(row['rank_final']):>10} | "
            f"{str(row['in_top_k']):>8}"
        )
    return "\n".join(lines)


def _fmt_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _demo_chunks() -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
    """Synthetic trace lists for the module demo (no Qdrant / LLM)."""
    retrieved = [
        Chunk("noise passage about unrelated topic", metadata={"doc_id": "distractor", "chunk_role": "small"}, score=0.91),
        Chunk("relevant passage about vitamin D", metadata={"doc_id": "gold_doc", "chunk_role": "small"}, score=0.88),
        Chunk("another distractor", metadata={"doc_id": "other", "chunk_role": "small"}, score=0.85),
    ]
    reranked = [
        Chunk("relevant passage about vitamin D", metadata={"doc_id": "gold_doc", "chunk_role": "small"}, score=0.97),
        Chunk("noise passage about unrelated topic", metadata={"doc_id": "distractor", "chunk_role": "small"}, score=0.42),
        Chunk("another distractor", metadata={"doc_id": "other", "chunk_role": "small"}, score=0.31),
    ]
    final = reranked[:2]
    return retrieved, reranked, final


def _print_overview() -> None:
    print(
        """rank_report — format & analyze replay query traces (no I/O)

Role
  Pure helpers used by ``_eval_.analysis.replay.query``. After ``aquery_trace`` returns,
  this module turns raw chunk lists into human-readable tables and gold-doc
  rank comparisons.

When to use
  * Diagnose where a relevant doc was lost: vector recall vs rerank vs top_k cut.
  * Compare ``semantic_rerank``: did reranking move ``gold_doc`` from #2 → #1?

Exports
  rank_of_doc          1-based doc rank in a deduplicated chunk list
  gold_rank_report     per-gold-doc ranks at retrieved / reranked / final
  format_stage_table   rank | doc_id | chunk_role | score | preview
  format_gold_table    gold doc_id | rank@* stages | in_top_k

Consumed by
  ``python -m _eval_.analysis.replay.query`` (CLI tables and --json gold_report)
"""
    )


def main() -> int:
    """Print module overview and a synthetic stage / gold-rank demo."""
    _print_overview()

    retrieved, reranked, final = _demo_chunks()
    print("=== Demo: format_stage_table (retrieved vs reranked) ===\n")
    print(format_stage_table(retrieved, stage="retrieved"))
    print()
    print(format_stage_table(reranked, stage="reranked"))
    print()

    rows = gold_rank_report(
        {"gold_doc", "missing_gold"},
        retrieved=retrieved,
        reranked=reranked,
        final=final,
        top_k=2,
    )
    print("=== Demo: format_gold_table (top_k=2) ===\n")
    print(format_gold_table(rows))
    print(
        "\nInterpretation: gold_doc rank improves 2→1 after rerank and stays in top_k;"
        " missing_gold never appears in any stage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
