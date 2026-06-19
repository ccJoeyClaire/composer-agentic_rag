"""Per-profile RAG eval: index the pooled docs, run queries, score them.

Profile flags are read directly from ``arg_config.yaml`` → ``profiles``; RAG
assembly uses :mod:`rag.build` without an intermediate wrapper.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

import yaml
from tqdm import tqdm

from rag.base import Chunk
from rag.build import build_RAG_indexer, build_RAG_retriever
from rag.core import RAGIndexer, RAGRetriever

from _eval_.config import RunConfig, collection_name
from _eval_.data_preparing.beir import (
    CorpusDoc,
    DocId,
    EvalQuery,
    Qrels,
    QueryId,
    resolve_chunk_doc_id,
)
from _eval_.data_preparing.prepare import PreparedEvalData
from _eval_.data_preparing.pooling import gold_docs
from _eval_.scoring.metrics import mean_metrics, query_metrics
from _eval_.paths import REPO_ROOT

# Keep in sync with rag/config.py _PROFILE_BOOL_FIELDS.
_PROFILE_BOOL_FIELDS = (
    "use_token_chunker",
    "use_contextual",
    "use_small_to_big",
    "use_predict_questions",
    "use_hyde",
    "use_reranker",
)

_ARG_CONFIG_PATH = REPO_ROOT / "arg_config.yaml"

# Duplicated in agent_eval.pipeline (independent eval lines). Keep in sync;
# parity tests: tests/eval/test_store_ops.py


async def drop_collection(store) -> None:
    """Delete the backing collection when it exists."""
    if await store.client.collection_exists(store.collection):
        await store.client.delete_collection(store.collection)


async def collection_exists(store) -> bool:
    """Return whether the store's backing collection is already present."""
    return await store.client.collection_exists(store.collection)


async def index_doc_list(
    indexer: RAGIndexer,
    docs: list[CorpusDoc],
    *,
    concurrency: int,
    desc: str = "index",
) -> int:
    """Index a list of docs with bounded concurrency; returns docs indexed."""
    if not docs:
        return 0

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(docs), desc=desc, unit="doc", dynamic_ncols=True)

    async def _index_one(doc: CorpusDoc) -> None:
        async with semaphore:
            source = doc.title or doc.doc_id
            await indexer.aindex(doc.text, source=source, doc_id=doc.doc_id)
            pbar.update(1)

    try:
        await asyncio.gather(*(_index_one(doc) for doc in docs))
    finally:
        pbar.close()
    return len(docs)


class QueryScore(TypedDict):
    query_id: str
    num_gold: int
    num_ranked_docs: int
    metrics: dict[str, float]


@dataclass
class ProfileResult:
    """Aggregate outcome for one RAG profile over the evaluated queries."""

    profile_id: str
    collection: str
    num_docs_indexed: int
    num_queries: int
    mean_metrics: dict[str, float]
    per_query: list[QueryScore]


def _load_profile_flags(profile_id: str) -> dict[str, bool]:
    """Read one profile's boolean flags from ``arg_config.yaml``."""
    with _ARG_CONFIG_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    profiles = data["profiles"]
    raw = profiles[profile_id]
    return {key: bool(raw.get(key, False)) for key in _PROFILE_BOOL_FIELDS}


def ranked_doc_ids(chunks: list[Chunk]) -> list[DocId]:
    """Map retrieved chunks to a deduplicated, rank-preserving doc id list."""
    seen: set[DocId] = set()
    ranked: list[DocId] = []
    for chunk in chunks:
        doc_id = resolve_chunk_doc_id(chunk.metadata)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
    return ranked


async def score_queries(
    retriever: RAGRetriever,
    queries: dict[QueryId, EvalQuery],
    qrels: Qrels,
    query_ids: list[QueryId],
    cfg: RunConfig,
    *,
    desc: str = "score",
) -> list[QueryScore]:
    """Run retrieval for each query and compute doc-level ranking metrics."""
    results: list[QueryScore] = []
    for qid in tqdm(query_ids, desc=desc, unit="query", dynamic_ncols=True):
        relevance = qrels[qid]
        gold = gold_docs(relevance, cfg.pool_spec.rel_threshold)
        chunks = await retriever.aquery(queries[qid].text, top_k=cfg.fetch_chunks)
        ranked = ranked_doc_ids(chunks)
        results.append(
            QueryScore(
                query_id=qid,
                num_gold=len(gold),
                num_ranked_docs=len(ranked),
                metrics=query_metrics(
                    ranked,
                    relevance,
                    gold,
                    k_values=cfg.k_values,
                    mrr_k=cfg.max_k,
                ),
            )
        )
    return results


async def eval_pipeline(
    profile_id: str,
    data: PreparedEvalData,
    cfg: RunConfig,
) -> ProfileResult:
    """Index the prepared pool for ``profile_id`` then score every query.

    Indexer and retriever share one store + embedder so search hits exactly the
    docs just indexed. The store is closed before returning so the next profile
    run can open a fresh client connection.
    """
    flags = _load_profile_flags(profile_id)
    collection = collection_name(cfg.dataset, profile_id)

    predict_concurrency = (
        cfg.predict_question_max_concurrency if flags["use_predict_questions"] else None
    )
    indexer = build_RAG_indexer(
        collection,
        use_token_chunker=flags["use_token_chunker"],
        use_contextual=flags["use_contextual"],
        use_predict_questions=flags["use_predict_questions"],
        use_small_to_big=flags["use_small_to_big"],
        predict_question_max_concurrency=predict_concurrency,
    )
    retriever = build_RAG_retriever(
        collection,
        use_reranker=flags["use_reranker"],
        use_contextual=flags["use_contextual"],
        use_hyde=flags["use_hyde"],
        use_small_to_big=flags["use_small_to_big"],
        store=indexer.store,
        embedder=indexer.embedder,
    )

    if cfg.recreate:
        await drop_collection(indexer.store)

    if cfg.recreate or not await collection_exists(indexer.store):
        num_docs = await index_doc_list(
            indexer,
            data.pool,
            concurrency=cfg.index_concurrency,
            desc=f"{profile_id} index",
        )
    else:
        num_docs = len(data.pool)
    per_query = await score_queries(
        retriever,
        data.queries,
        data.qrels,
        data.query_ids,
        cfg,
        desc=f"{profile_id} score",
    )
    await indexer.store.aclose()

    return ProfileResult(
        profile_id=profile_id,
        collection=collection,
        num_docs_indexed=num_docs,
        num_queries=len(per_query),
        mean_metrics=mean_metrics([q["metrics"] for q in per_query]),
        per_query=per_query,
    )
