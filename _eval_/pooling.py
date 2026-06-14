"""Pooled-subset selection: decide which corpus docs to index for a run.

Indexing a full BEIR corpus (trec-covid has ~171k docs, msmarco ~8.8M) is slow
and expensive to embed. For retrieval metrics we only strictly need, per query:

* every judged-relevant doc (so recall is achievable at all), plus
* a bounded number of judged-irrelevant "distractor" docs (so ranking is a real
  task, not a trivial one).

:func:`build_pool` returns exactly that set of doc ids. The :class:`PoolSpec`
knobs are the extension point for future strategies (random unjudged negatives,
hard-negative mining, per-query caps, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

from _eval_.beir import DocId, QueryId, Qrels


@dataclass(frozen=True)
class PoolSpec:
    """Parameters controlling pooled-subset size and what counts as gold.

    Attributes:
        rel_threshold: A judged score ``>= rel_threshold`` counts as gold/relevant.
            trec-covid (0/1/2): use 1 for lenient, 2 for strict. msmarco: 1.
        max_distractors_per_query: Cap on judged-irrelevant docs kept per query
            to bound index size. ``None`` keeps every judged doc.
    """

    rel_threshold: int = 1
    max_distractors_per_query: int | None = None


def gold_docs(relevance: dict[DocId, int], rel_threshold: int) -> set[DocId]:
    """Return doc ids judged relevant for one query (score >= threshold)."""
    return {doc_id for doc_id, score in relevance.items() if score >= rel_threshold}


def queries_with_gold(qrels: Qrels, rel_threshold: int) -> list[QueryId]:
    """Query ids that have at least one gold doc; recall is undefined otherwise.

    Sorted by (id length, id) so a numeric-id dataset yields a stable, intuitive
    order (1, 2, ... 10) and a ``--limit`` prefix is reproducible.
    """
    qualifying = [
        qid for qid, rel in qrels.items() if gold_docs(rel, rel_threshold)
    ]
    return sorted(qualifying, key=lambda q: (len(q), q))


def build_pool(
    qrels: Qrels,
    query_ids: list[QueryId],
    spec: PoolSpec,
) -> set[DocId]:
    """Collect the doc ids to index for ``query_ids``.

    Args:
        qrels: Full graded qrels for the dataset.
        query_ids: The queries actually being evaluated.
        spec: Pooling parameters (threshold + distractor cap).

    Returns:
        Union over the selected queries of their gold docs plus (capped)
        judged-irrelevant distractor docs.
    """
    pool: set[DocId] = set()
    for qid in query_ids:
        relevance = qrels.get(qid, {})
        pool |= gold_docs(relevance, spec.rel_threshold)

        distractors = [
            doc_id
            for doc_id, score in relevance.items()
            if score < spec.rel_threshold
        ]
        if spec.max_distractors_per_query is not None:
            distractors = distractors[: spec.max_distractors_per_query]
        pool.update(distractors)
    return pool
