"""LLM-facing RAG tools. Deployment binding lives in :mod:`rag.context`.

Run (from repo root):
  python -m rag.build index --collection demo --in-memory
  python -m rag.build search --collection demo --in-memory "your query"
"""

from __future__ import annotations

import asyncio
from typing import Annotated, List, Optional

from pydantic import Field

from rag.context import (
    RagToolContext,
    bind_indexer,
    bind_rag_context,
    bind_retriever,
    get_active_context,
)
from tools.registry import local_tool

__all__ = [
    "RagToolContext",
    "RAG_index_tool",
    "RAG_search_tool",
    "bind_indexer",
    "bind_rag_context",
    "bind_retriever",
    "get_active_context",
]


def _run_async(coro):
    """Run coroutine from sync tool entrypoints (works inside asyncio loops too)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _append_notes(body: str, notes: List[str]) -> str:
    if not notes:
        return body
    note_block = "\n".join(f"[note] {note}" for note in notes)
    return f"{body}\n\n{note_block}" if body else note_block


@local_tool
def RAG_index_tool(
    text: Annotated[str, Field(description="待入库正文")],
    source: Annotated[str, Field(description="文档来源标识")],
    use_predict_questions: Annotated[
        bool,
        Field(description="是否为每个 chunk 生成预设问题以增强召回（仅影响索引，不影响检索结构）"),
    ] = False,
) -> str:
    """Index document text into the knowledge base."""
    ctx = get_active_context()
    indexer, notes = ctx.resolve_indexer(use_predict_questions=use_predict_questions)
    if indexer is None:
        return "RAG indexer not bound. Call bind_rag_context() or bind_indexer() at startup."
    ok = _run_async(indexer.aindex(text, source=source))
    body = (
        f"Successfully indexed document '{source}'."
        if ok
        else f"Failed to index or verify document '{source}'."
    )
    return _append_notes(body, notes)


@local_tool
def RAG_search_tool(
    query: Annotated[str, Field(description="检索问题")],
    use_hyde: Annotated[
        bool, Field(description="是否用 HyDE 改写查询向量（查询期，可运行时切换）")
    ] = False,
    use_reranker: Annotated[
        bool, Field(description="是否挂载 CrossEncoder 精排（查询期，可运行时切换）")
    ] = False,
    recall_n: Annotated[
        Optional[int], Field(description="rerank 前的向量召回条数，留空用默认", ge=1)
    ] = None,
    top_k: Annotated[
        Optional[int], Field(description="返回的 chunk 数，留空用默认", ge=1)
    ] = None,
) -> str:
    """Search the knowledge base and return relevant context.

    Query-time modes (use_hyde / use_reranker / recall_n / top_k) are selectable
    per call within the deployment's allowed range; index-coupled modes are fixed.
    """
    ctx = get_active_context()
    retriever, notes = ctx.resolve_retriever(
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        recall_n=recall_n,
    )
    if retriever is None:
        return "RAG retriever not bound. Call bind_rag_context() or bind_retriever() at startup."

    effective_top_k = top_k if top_k is not None else ctx.default_top_k
    chunks = _run_async(retriever.aquery(query, top_k=effective_top_k))
    body = "\n\n---\n\n".join(c.content for c in chunks)
    return _append_notes(body, notes)
