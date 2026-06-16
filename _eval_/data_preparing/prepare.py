"""Unified data preparation: load BEIR files and build the pooled doc subset."""

from __future__ import annotations

from dataclasses import dataclass

from _eval_.config import DATASETS, RunConfig
from _eval_.data_preparing.beir import (
    CorpusDoc,
    EvalQuery,
    Qrels,
    QueryId,
    iter_corpus,
    load_qrels,
    load_queries,
)
from _eval_.data_preparing.pooling import build_pool, queries_with_gold


@dataclass(frozen=True)
class PreparedEvalData:
    """Everything downstream eval pipelines need after one prepare pass."""

    pool: list[CorpusDoc]
    queries: dict[QueryId, EvalQuery]
    qrels: Qrels
    query_ids: list[QueryId]


def prepare_eval_data(cfg: RunConfig) -> PreparedEvalData:
    """Load qrels/queries, pick evaluable queries, materialize the doc pool.

    Args:
        cfg: Typed run configuration from :func:`_eval_.config.load_rag_eval_config`.

    Returns:
        Prepared structures ready for indexing and scoring.

    Raises:
        KeyError: Unknown ``cfg.dataset`` id in :data:`DATASETS`.
        SystemExit: No evaluable queries after filtering.
    """
    spec = DATASETS[cfg.dataset]
    qrels = load_qrels(spec.qrels_path())
    queries = load_queries(spec.queries_path())

    candidate_ids = queries_with_gold(qrels, cfg.pool_spec.rel_threshold)
    query_ids = [qid for qid in candidate_ids if qid in queries]
    if cfg.query_limit is not None:
        query_ids = query_ids[: cfg.query_limit]
    if not query_ids:
        raise SystemExit("No evaluable queries (no gold under threshold / id mismatch).")

    pool_ids = build_pool(qrels, query_ids, cfg.pool_spec)
    pool = list(iter_corpus(spec.corpus_path(), keep_ids=pool_ids))
    return PreparedEvalData(
        pool=pool,
        queries=queries,
        qrels=qrels,
        query_ids=query_ids,
    )
