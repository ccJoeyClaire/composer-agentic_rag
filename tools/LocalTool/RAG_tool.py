from __future__ import annotations

import asyncio
from typing import Annotated, Optional

from pydantic import Field

from rag.core import RAGIndexer, RAGRetriever
from tools.registry import local_tool

_indexer: Optional[RAGIndexer] = None
_retriever: Optional[RAGRetriever] = None
_search_top_k: int = 5


def bind_indexer(indexer: RAGIndexer) -> None:
    global _indexer
    _indexer = indexer


def bind_retriever(retriever: RAGRetriever, *, top_k: int = 5) -> None:
    global _retriever, _search_top_k
    _retriever = retriever
    _search_top_k = top_k


def _run_async(coro):
    """Run coroutine from sync tool entrypoints (works inside asyncio loops too)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


@local_tool
def RAG_index_tool(
    text: Annotated[str, Field(description="待入库正文")],
    source: Annotated[str, Field(description="文档来源标识")],
) -> str:
    """Index document text into the knowledge base."""
    if _indexer is None:
        return "RAG indexer not bound. Call bind_indexer() at startup."
    ok = _run_async(_indexer.aindex(text, source=source))
    if ok:
        return f"Successfully indexed document '{source}'."
    return f"Failed to index or verify document '{source}'."


@local_tool
def RAG_search_tool(
    query: Annotated[str, Field(description="检索问题")],
) -> str:
    """Search the knowledge base and return relevant context."""
    if _retriever is None:
        return "RAG retriever not bound. Call bind_retriever() at startup."
    chunks = _run_async(_retriever.aquery(query, top_k=_search_top_k))
    return "\n\n---\n\n".join(c.content for c in chunks)
