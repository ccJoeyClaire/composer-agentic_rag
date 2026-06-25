"""Ranking metrics over a deduplicated, doc-level ranked list.

All functions take an already-ranked list of unique ``DocId`` (best first) plus
the query's graded relevance, so they stay pure and trivial to unit-test. The
retrieval side (chunks -> ranked doc ids) lives in :mod:`_eval_.rag_eval.pipeline`.

Definitions (k = cutoff):
* Recall@k — fraction of the query's gold docs found in the top-k.
* MRR@k    — reciprocal rank of the first gold doc within top-k (else 0).
* nDCG@k   — graded DCG normalized by the ideal DCG, rewarding higher scores
             ranked earlier.
"""

from __future__ import annotations

import math

from _eval_.data_preparing.beir import DocId, RelevanceMap


def recall_at_k(ranked: list[DocId], gold: set[DocId], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in gold)
    return hits / len(gold)


def hit_at_k(ranked: list[DocId], gold: set[DocId], k: int) -> float:
    """Binary success@k: 1.0 if any gold doc is in the top-k, else 0.0."""
    return 1.0 if any(doc_id in gold for doc_id in ranked[:k]) else 0.0


def mrr_at_k(ranked: list[DocId], gold: set[DocId], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def _dcg(gains: list[float]) -> float:
    # Standard DCG: position 1 has denominator log2(2) == 1.
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked: list[DocId], relevance: RelevanceMap, k: int) -> float:
    gains = [float(relevance.get(doc_id, 0)) for doc_id in ranked[:k]]
    ideal_gains = sorted((float(s) for s in relevance.values()), reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    if idcg == 0.0:
        return 0.0
    return _dcg(gains) / idcg


def query_metrics(
    ranked: list[DocId],
    relevance: RelevanceMap,
    gold: set[DocId],
    *,
    k_values: tuple[int, ...],
    mrr_k: int,
) -> dict[str, float]:
    """Compute the full metric set for one query as a flat ``name -> value`` map.

    Keys look like ``recall@10`` / ``ndcg@10`` / ``hit@10`` / ``mrr@20`` so they
    can be averaged across queries and rendered as table columns directly.
    """
    scores: dict[str, float] = {}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked, relevance, k)
        scores[f"hit@{k}"] = hit_at_k(ranked, gold, k)
    scores[f"mrr@{mrr_k}"] = mrr_at_k(ranked, gold, mrr_k)
    return scores


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric across queries; empty input yields an empty map."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: sum(q[key] for q in per_query) / n for key in keys}
