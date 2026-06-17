"""Streaming loaders and core types for BEIR-format datasets.

A BEIR dataset is three files that join on ids:

* ``corpus.jsonl``  — ``{"_id", "title", "text", ...}`` (the knowledge base).
* ``queries.jsonl`` — ``{"_id", "text", ...}`` (the questions).
* ``qrels/*.tsv``   — ``query-id <tab> corpus-id <tab> score`` (the gold labels;
  ``score`` is GRADED, e.g. trec-covid uses 0/1/2, msmarco uses 1).

The corpus can be millions of lines, so :func:`iter_corpus` streams and can
filter to a ``keep_ids`` set (used by the pooled-subset step) without loading
the whole file into memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# One canonical name per concept, reused across the package.
DocId = str
QueryId = str
# query_id -> {doc_id -> graded relevance score (>=0; 0 == judged not relevant)}.
RelevanceMap = dict[DocId, int]
Qrels = dict[QueryId, RelevanceMap]

# Chunk metadata keys stamped at index time (see rag/core.py ``aindex``).
SOURCE_META_KEY = "source"
DOC_ID_META_KEY = "doc_id"

_DEFAULT_ID_FIELD = "_id"
_DEFAULT_TEXT_FIELD = "text"
_DEFAULT_TITLE_FIELD = "title"


@dataclass(frozen=True)
class CorpusDoc:
    """One BEIR corpus row: body text for chunking plus optional title metadata."""

    doc_id: DocId
    text: str
    title: str = ""


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query."""

    query_id: QueryId
    text: str


def resolve_chunk_doc_id(metadata: dict[str, object] | None) -> DocId | None:
    """Map stored chunk metadata back to a BEIR corpus id for qrels scoring."""
    meta = metadata or {}
    doc_id = meta.get(DOC_ID_META_KEY)
    if doc_id:
        return str(doc_id)
    legacy = meta.get(SOURCE_META_KEY)
    return str(legacy) if legacy else None


def iter_corpus(
    path: Path,
    *,
    keep_ids: set[DocId] | None = None,
    id_field: str = _DEFAULT_ID_FIELD,
    text_field: str = _DEFAULT_TEXT_FIELD,
    title_field: str = _DEFAULT_TITLE_FIELD,
) -> Iterator[CorpusDoc]:
    """Stream ``corpus.jsonl`` as :class:`CorpusDoc` records.

    Args:
        path: Path to the ``corpus.jsonl`` file.
        keep_ids: When given, yield only docs whose id is in this set and stop
            early once every requested id has been emitted. This keeps indexing
            cost bounded for huge corpora.
        id_field/text_field/title_field: Field name overrides for non-standard
            BEIR exports.

    Yields:
        :class:`CorpusDoc` with ``text`` set to the corpus body only; ``title`` is
        kept separate for contextual indexing (``source`` at index time).
    """
    remaining = set(keep_ids) if keep_ids is not None else None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            doc_id = str(raw.get(id_field, "")).strip()
            if not doc_id:
                continue
            if remaining is not None:
                if doc_id not in remaining:
                    continue
                remaining.discard(doc_id)
            title = str(raw.get(title_field, "") or "").strip()
            body = str(raw.get(text_field, "") or "").strip()
            yield CorpusDoc(doc_id=doc_id, text=body, title=title)
            if remaining is not None and not remaining:
                break


def load_queries(
    path: Path,
    *,
    id_field: str = _DEFAULT_ID_FIELD,
    text_field: str = _DEFAULT_TEXT_FIELD,
) -> dict[QueryId, EvalQuery]:
    """Load ``queries.jsonl`` into a ``query_id -> EvalQuery`` map."""
    queries: dict[QueryId, EvalQuery] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            qid = str(raw.get(id_field, "")).strip()
            if not qid:
                continue
            queries[qid] = EvalQuery(query_id=qid, text=str(raw.get(text_field, "")))
    return queries


def load_qrels(path: Path) -> Qrels:
    """Load a BEIR qrels TSV, preserving graded scores.

    The header row (``query-id corpus-id score``) and any malformed lines are
    skipped because their score column does not parse as an int.
    """
    qrels: Qrels = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, raw_score = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                score = int(raw_score)
            except ValueError:
                continue  # header or junk line
            if not qid or not doc_id:
                continue
            qrels.setdefault(qid, {})[doc_id] = score
    return qrels


def _smoke_dataset_id() -> str:
    return "trec-covid"


def _preview_text(text: str, max_chars: int = 200) -> str:
    """Truncate long corpus bodies for terminal smoke output."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... ({len(text)} chars total)"


def _print_section(title: str, fields: dict[str, object]) -> None:
    print(f"\n--- {title} ---")
    for key, value in fields.items():
        print(f"  {key}: {value}")


def _main() -> None:
    """Print sample structures from the configured smoke dataset."""
    from _eval_.config import DATASETS

    dataset_id = _smoke_dataset_id()
    spec = DATASETS[dataset_id]
    print(f"BEIR smoke load: {dataset_id}")

    corpus_path = spec.corpus_path()
    first_doc = next(iter_corpus(corpus_path), None)
    if first_doc is not None:
        _print_section(
            "Corpus (first doc)",
            {
                "doc_id": first_doc.doc_id,
                "title": first_doc.title,
                "text": _preview_text(first_doc.text),
            },
        )

    queries = load_queries(spec.queries_path())
    first_qid = next(iter(queries), None)
    if first_qid is not None:
        query = queries[first_qid]
        _print_section(
            "Query (first)",
            {"query_id": query.query_id, "text": query.text},
        )

    qrels = load_qrels(spec.qrels_path())
    first_rel_qid = next(iter(qrels), None)
    if first_rel_qid is not None:
        rel = qrels[first_rel_qid]
        first_doc_id = next(iter(rel), None)
        _print_section(
            "Qrels (first row)",
            {
                "query_id": first_rel_qid,
                "doc_id": first_doc_id,
                "score": rel.get(first_doc_id),
            },
        )

    _print_section(
        "Counts",
        {"queries": len(queries), "qrels_queries": len(qrels)},
    )
    print()


if __name__ == "__main__":
    _main()
