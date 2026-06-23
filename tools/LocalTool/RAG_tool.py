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
from rag.profile_schema import (
    TOP_K_KEY,
    index_profile_from_optional_args,
    search_profile_from_optional_args,
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
    use_token_chunker: Annotated[
        bool | None,
        Field(description="是否用 TokenChunker 硬切（留空用部署默认 profile）"),
    ] = None,
    use_contextual: Annotated[
        bool | None,
        Field(description="是否启用 ContextualEnricher（留空用部署默认 profile）"),
    ] = None,
    use_small_to_big: Annotated[
        bool | None,
        Field(description="是否启用 small-to-big 索引（留空用部署默认 profile）"),
    ] = None,
    use_predict_questions: Annotated[
        bool | None,
        Field(description="是否为每个 chunk 生成预设问题（留空用部署默认 profile）"),
    ] = None,
) -> str:
    """Index document text into the knowledge base."""
    ctx = get_active_context()
    profile = index_profile_from_optional_args(
        use_token_chunker=use_token_chunker,
        use_contextual=use_contextual,
        use_small_to_big=use_small_to_big,
        use_predict_questions=use_predict_questions,
    )
    indexer, notes, _ = ctx.resolve_indexer(profile)
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
    use_contextual: Annotated[
        bool | None,
        Field(description="查询期是否拼接 contextual header（留空用部署默认 profile）"),
    ] = None,
    use_small_to_big: Annotated[
        bool | None,
        Field(description="查询期是否 small-to-big 召回 parent（留空用部署默认 profile）"),
    ] = None,
    use_hyde: Annotated[
        bool | None,
        Field(description="是否用 HyDE 改写查询向量（留空用部署默认 profile）"),
    ] = None,
    use_reranker: Annotated[
        bool | None,
        Field(description="是否挂载 CrossEncoder 精排（留空用部署默认 profile）"),
    ] = None,
    recall_n: Annotated[
        Optional[int],
        Field(description="rerank 前的向量召回条数，留空用部署默认", ge=1),
    ] = None,
    top_k: Annotated[
        Optional[int],
        Field(description="返回的 chunk 数，留空用部署默认", ge=1),
    ] = None,
) -> str:
    """Search the knowledge base and return relevant context.

    All pipeline bools are selectable per call within the deployment allow-range.
    Omitted args fall back to the bound profile defaults (``baseline`` by default).
    Index and search options need not match.
    """
    ctx = get_active_context()
    profile = search_profile_from_optional_args(
        use_contextual=use_contextual,
        use_small_to_big=use_small_to_big,
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        recall_n=recall_n,
        top_k=top_k,
    )
    retriever, notes, eff = ctx.resolve_retriever(profile)
    if retriever is None:
        return "RAG retriever not bound. Call bind_rag_context() or bind_retriever() at startup."

    chunks = _run_async(retriever.aquery(query, top_k=eff[TOP_K_KEY]))
    body = "\n\n---\n\n".join(c.content for c in chunks)
    return _append_notes(body, notes)
