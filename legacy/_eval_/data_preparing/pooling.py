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

from _eval_.data_preparing.beir import DocId, QueryId, Qrels


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


def distractor_docs(
    relevance: dict[DocId, int],
    rel_threshold: int,
    max_distractors: int | None = None,
) -> set[DocId]:
    """Return judged-irrelevant doc ids for one query (score < threshold).

    When ``max_distractors`` is set, only the first N in qrels iteration order
    are kept so pool size stays bounded without changing which docs count as gold.
    """
    distractors = [
        doc_id for doc_id, score in relevance.items() if score < rel_threshold
    ]
    if max_distractors is not None:
        distractors = distractors[:max_distractors]
    return set(distractors)


def queries_with_gold(qrels: Qrels, rel_threshold: int) -> list[QueryId]:
    """Query ids that have at least one gold doc; recall is undefined otherwise.

    Sorted by (id length, id) so a numeric-id dataset yields a stable, intuitive
    order (1, 2, ... 10) and a limit prefix is reproducible.
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
        pool |= distractor_docs(
            relevance, spec.rel_threshold, spec.max_distractors_per_query
        )
    return pool


def _main() -> None:
    """Print pool stats for a small query subset of the smoke dataset."""
    from _eval_.config import DATASETS, DEFAULT_QUERY_LIMIT
    from _eval_.data_preparing.beir import load_qrels

    dataset_id = "trec-covid"
    spec = DATASETS[dataset_id]
    rel_threshold = 1
    qrels = load_qrels(spec.qrels_path())

    query_ids = queries_with_gold(qrels, rel_threshold)[:DEFAULT_QUERY_LIMIT]
    pool_spec = PoolSpec(rel_threshold=rel_threshold, max_distractors_per_query=100)
    pool_ids = build_pool(qrels, query_ids, pool_spec)

    gold_counts = [len(gold_docs(qrels[qid], rel_threshold)) for qid in query_ids]
    print(f"dataset={dataset_id} query_ids={query_ids}")
    print(f"pool_docs={len(pool_ids)} gold_per_query={gold_counts}")
    print(f"pool_spec={pool_spec!r}")


if __name__ == "__main__":
    _main()
