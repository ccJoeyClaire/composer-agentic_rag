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

# Metadata key the indexer stamps the source doc id under (see rag/core.py).
SOURCE_META_KEY = "source"

_DEFAULT_ID_FIELD = "_id"
_DEFAULT_TEXT_FIELD = "text"
_DEFAULT_TITLE_FIELD = "title"


@dataclass(frozen=True)
class CorpusDoc:
    """One indexable document after title+body composition."""

    doc_id: DocId
    text: str
    title: str = ""


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query."""

    query_id: QueryId
    text: str


def _compose_text(title: str, body: str) -> str:
    """Join a BEIR title and body the same way the legacy adapter does."""
    title, body = title.strip(), body.strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


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
        :class:`CorpusDoc` with ``text`` already title+body composed.
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
            title = str(raw.get(title_field, "") or "")
            body = str(raw.get(text_field, "") or "")
            yield CorpusDoc(doc_id=doc_id, text=_compose_text(title, body), title=title)
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
