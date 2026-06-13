"""
Index documents from a dataset manifest into Qdrant for a given RAG profile.

Usage (repo root):
  python -m eval.rag.run_index --dataset smoke --profile baseline
  python -m eval.rag.run_index --dataset smoke --profile full --recreate
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Literal, TypedDict

from eval.bootstrap import setup_eval_env

setup_eval_env()

from eval.loaders import load_documents
from eval.profiles import (
    RAGProfile,
    build_indexer_for_profile,
    collection_name,
    index_concurrency,
)


class IndexDocumentOk(TypedDict):
    source: str
    ok: bool
    chunk_count: int


class IndexDocumentError(TypedDict):
    ok: Literal[False]
    error: str


IndexDocumentDetail = IndexDocumentOk | IndexDocumentError


class IndexDatasetSummary(TypedDict):
    dataset: str
    profile: str
    collection: str
    total_documents: int
    total_chunks: int
    ok: bool
    details: list[IndexDocumentDetail]


async def _recreate_collection(indexer) -> None:
    store = indexer.store
    client = store.client
    if await client.collection_exists(store.collection):
        await client.delete_collection(store.collection)


async def index_dataset(
    *,
    dataset: str,
    profile: RAGProfile,
    in_memory: bool,
    recreate: bool,
) -> IndexDatasetSummary:
    collection = collection_name(dataset, profile.profile_id)
    indexer = build_indexer_for_profile(profile, collection, in_memory=in_memory)

    if recreate:
        await _recreate_collection(indexer)

    documents = load_documents(dataset)
    sem = asyncio.Semaphore(index_concurrency())
    results: list[IndexDocumentDetail] = []

    async def _index_one(doc_id: str, text: str) -> IndexDocumentOk:
        async with sem:
            ok = await indexer.aindex(text, source=doc_id)
            count = await indexer.store.acount_by_source(doc_id)
            return IndexDocumentOk(source=doc_id, ok=ok, chunk_count=count)

    tasks = []
    for doc in documents:
        tasks.append(_index_one(doc["doc_id"], doc["text"]))

    indexed = await asyncio.gather(*tasks, return_exceptions=True)
    for item in indexed:
        if isinstance(item, Exception):
            results.append(IndexDocumentError(ok=False, error=str(item)))
        else:
            results.append(item)

    await indexer.store.aclose()

    ok_all = all(detail.get("ok") for detail in results if "ok" in detail)
    total_chunks = sum(
        detail["chunk_count"] for detail in results if "chunk_count" in detail
    )
    return IndexDatasetSummary(
        dataset=dataset,
        profile=profile.profile_id,
        collection=collection,
        total_documents=len(documents),
        total_chunks=total_chunks,
        ok=ok_all,
        details=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index eval dataset manifest into Qdrant.")
    parser.add_argument("--dataset", default="smoke", help="Dataset name under eval/datasets/")
    parser.add_argument("--profile", required=True, help="RAG profile id (baseline, contextual, ...)")
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use ephemeral local Qdrant instead of Docker (default: Docker 127.0.0.1:6333)",
    )
    parser.add_argument("--recreate", action="store_true", help="Delete collection before indexing")
    args = parser.parse_args()

    profile = RAGProfile.get(args.profile)
    summary = asyncio.run(
        index_dataset(
            dataset=args.dataset,
            profile=profile,
            in_memory=args.in_memory,
            recreate=args.recreate,
        )
    )
    print(summary)
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
