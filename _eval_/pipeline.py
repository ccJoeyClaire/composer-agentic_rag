"""Per-profile evaluation: index the pooled docs, run queries, score them.

Reuses the existing profile wiring in :mod:`eval.profiles` (which reads
``arg_config.yaml`` and assembles the chunker/embedder/store/retriever for a
profile) so this package only owns the eval-specific logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

from eval.profiles import (
    RAGProfile,
    build_indexer_for_profile,
    build_retriever_for_profile,
)
from rag.base import Chunk
from rag.core import RAGIndexer, RAGRetriever

from _eval_.beir import (
    CorpusDoc,
    DocId,
    EvalQuery,
    Qrels,
    QueryId,
    SOURCE_META_KEY,
    iter_corpus,
)
from _eval_.config import RunConfig, collection_name
from _eval_.metrics import mean_metrics, query_metrics
from _eval_.pooling import gold_docs


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


def _ranked_doc_ids(chunks: list[Chunk]) -> list[DocId]:
    """Map retrieved chunks to a deduplicated, rank-preserving doc id list.

    Several chunks can come from the same source document; doc-level metrics
    need each doc counted once at its best (first) rank.
    """
    seen: set[DocId] = set()
    ranked: list[DocId] = []
    for chunk in chunks:
        doc_id = (chunk.metadata or {}).get(SOURCE_META_KEY)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
    return ranked


async def drop_collection(store) -> None:
    if await store.client.collection_exists(store.collection):
        await store.client.delete_collection(store.collection)


async def index_doc_list(
    indexer: RAGIndexer,
    docs: list[CorpusDoc],
    *,
    concurrency: int,
) -> int:
    """Index a list of docs with bounded concurrency; returns docs indexed."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _index_one(text: str, source: str) -> None:
        async with semaphore:
            await indexer.aindex(text, source=source)

    await asyncio.gather(*(_index_one(doc.text, doc.doc_id) for doc in docs))
    return len(docs)


async def _score_queries(
    retriever: RAGRetriever,
    queries: dict[QueryId, EvalQuery],
    qrels: Qrels,
    query_ids: list[QueryId],
    cfg: RunConfig,
) -> list[QueryScore]:
    results: list[QueryScore] = []
    for qid in query_ids:
        relevance = qrels[qid]
        gold = gold_docs(relevance, cfg.pool_spec.rel_threshold)
        chunks = await retriever.aquery(queries[qid].text, top_k=cfg.fetch_chunks)
        ranked = _ranked_doc_ids(chunks)
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


async def evaluate_profile(
    profile: RAGProfile,
    *,
    corpus_path,
    pool_ids: set[DocId],
    queries: dict[QueryId, EvalQuery],
    qrels: Qrels,
    query_ids: list[QueryId],
    cfg: RunConfig,
) -> ProfileResult:
    """Index the pool for ``profile`` then score every evaluated query.

    Indexer and retriever share one store + embedder so search hits exactly the
    docs just indexed. The store is closed before returning so the next profile
    can reopen the local Qdrant path cleanly.
    """
    collection = collection_name(cfg.dataset, profile.profile_id)
    indexer = build_indexer_for_profile(profile, collection, in_memory=cfg.in_memory)

    if cfg.recreate:
        await drop_collection(indexer.store)

    docs = list(iter_corpus(corpus_path, keep_ids=pool_ids))
    num_docs = await index_doc_list(indexer, docs, concurrency=cfg.index_concurrency)

    retriever = build_retriever_for_profile(
        profile,
        collection,
        in_memory=cfg.in_memory,
        store=indexer.store,
        embedder=indexer.embedder,
    )
    per_query = await _score_queries(retriever, queries, qrels, query_ids, cfg)
    await indexer.store.aclose()

    return ProfileResult(
        profile_id=profile.profile_id,
        collection=collection,
        num_docs_indexed=num_docs,
        num_queries=len(per_query),
        mean_metrics=mean_metrics([q["metrics"] for q in per_query]),
        per_query=per_query,
    )
