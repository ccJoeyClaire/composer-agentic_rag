"""LLM-facing RAG tools. Connections via :data:`RAG_CONTEXT_KEY`; profiles are separate inject keys."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from rag.build import build_RAG_indexer, build_RAG_retriever
from rag.config import get_rag_config
from rag.context import (
    RAG_CONTEXT_KEY,
    RAG_INDEX_PROFILE_KEY,
    RAG_SEARCH_PROFILE_KEY,
    RagToolContext,
)
from rag.profile_schema import (
    RECALL_N_KEY,
    TOP_K_KEY,
    USE_CONTEXTUAL_KEY,
    USE_HYDE_KEY,
    USE_PREDICT_QUESTIONS_KEY,
    USE_RERANKER_KEY,
    USE_SMALL_TO_BIG_KEY,
    USE_TOKEN_CHUNKER_KEY,
    index_profile_from_optional_args,
    normalize_index_profile,
    normalize_search_profile,
    search_profile_from_optional_args,
)
from rag.serialize import chunks_to_tool_json
from tools.context import ToolContextError, ToolContextView
from tools.registry import local_tool

__all__ = [
    "RAG_CONTEXT_KEY",
    "RAG_index_tool",
    "RAG_search_tool",
    "RagToolContext",
]

_MISSING_RAG = "RAG context not bound. Call bind_rag() on ToolBox.context before invoke."
_MISSING_SEARCH_PROFILE = (
    "RAG search profile not bound. Call bind_rag() on ToolBox.context before invoke."
)
_MISSING_INDEX_PROFILE = (
    "RAG index profile not bound. Call bind_rag() on ToolBox.context before invoke."
)
_UNBOUND_STORE = "RAG store/embedder not bound. Call bind_rag() on ToolBox.context at startup."


def _append_notes(body: str, notes: list[str]) -> str:
    if not notes:
        return body
    note_block = "\n".join(f"[note] {note}" for note in notes)
    return f"{body}\n\n{note_block}" if body else note_block


def _require_ctx(
    tool_context: ToolContextView | None,
    key: str,
    typ: type,
    *,
    missing: str,
) -> object | str:
    if tool_context is None:
        return missing
    try:
        return tool_context.require(key, typ)
    except ToolContextError:
        return missing


@local_tool(context_keys=(RAG_CONTEXT_KEY, RAG_INDEX_PROFILE_KEY))
async def RAG_index_tool(
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
    *,
    _tool_context: ToolContextView | None = None,
) -> str:
    """Index document text into the knowledge base."""
    ctx = _require_ctx(
        _tool_context, RAG_CONTEXT_KEY, RagToolContext, missing=_MISSING_RAG
    )
    if isinstance(ctx, str):
        return ctx
    defaults = _require_ctx(
        _tool_context,
        RAG_INDEX_PROFILE_KEY,
        dict,
        missing=_MISSING_INDEX_PROFILE,
    )
    if isinstance(defaults, str):
        return defaults

    overrides = index_profile_from_optional_args(
        use_token_chunker=use_token_chunker,
        use_contextual=use_contextual,
        use_small_to_big=use_small_to_big,
        use_predict_questions=use_predict_questions,
    )
    eff, notes = normalize_index_profile(
        overrides,
        defaults=defaults,  # type: ignore[arg-type]
    )

    rag_ctx: RagToolContext = ctx  # type: ignore[assignment]
    if rag_ctx.store is None or rag_ctx.embedder is None:
        return _UNBOUND_STORE

    indexer = build_RAG_indexer(
        rag_ctx.collection,
        in_memory=rag_ctx.in_memory,
        use_token_chunker=bool(eff.get(USE_TOKEN_CHUNKER_KEY)),
        use_contextual=bool(eff.get(USE_CONTEXTUAL_KEY)),
        use_predict_questions=bool(eff.get(USE_PREDICT_QUESTIONS_KEY)),
        use_small_to_big=bool(eff.get(USE_SMALL_TO_BIG_KEY)),
        store=rag_ctx.store,
        embedder=rag_ctx.embedder,
    )
    ok = await indexer.aindex(text, source=source)
    body = (
        f"Successfully indexed document '{source}'."
        if ok
        else f"Failed to index or verify document '{source}'."
    )
    return _append_notes(body, notes)


@local_tool(context_keys=(RAG_CONTEXT_KEY, RAG_SEARCH_PROFILE_KEY))
async def RAG_search_tool(
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
        Field(description="是否挂载 CrossEncoder 精排（留空用 deployment 默认 profile）"),
    ] = None,
    recall_n: Annotated[
        int | None,
        Field(description="rerank 前的向量召回条数，留空用部署默认", ge=1),
    ] = None,
    top_k: Annotated[
        int | None,
        Field(description="返回的 chunk 数，留空用部署默认", ge=1),
    ] = None,
    *,
    _tool_context: ToolContextView | None = None,
) -> str:
    """Search the knowledge base and return relevant context."""
    ctx = _require_ctx(
        _tool_context, RAG_CONTEXT_KEY, RagToolContext, missing=_MISSING_RAG
    )
    if isinstance(ctx, str):
        return ctx
    defaults = _require_ctx(
        _tool_context,
        RAG_SEARCH_PROFILE_KEY,
        dict,
        missing=_MISSING_SEARCH_PROFILE,
    )
    if isinstance(defaults, str):
        return defaults

    overrides = search_profile_from_optional_args(
        use_contextual=use_contextual,
        use_small_to_big=use_small_to_big,
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        recall_n=recall_n,
        top_k=top_k,
    )
    retriever_cfg = get_rag_config().retriever
    eff, notes = normalize_search_profile(
        overrides,
        defaults=defaults,  # type: ignore[arg-type]
        max_recall_n=retriever_cfg.recall_n,
        max_top_k=None,
    )

    rag_ctx: RagToolContext = ctx  # type: ignore[assignment]
    if rag_ctx.store is None or rag_ctx.embedder is None:
        return _UNBOUND_STORE

    retriever = build_RAG_retriever(
        rag_ctx.collection,
        in_memory=rag_ctx.in_memory,
        use_reranker=bool(eff.get(USE_RERANKER_KEY)),
        use_contextual=bool(eff.get(USE_CONTEXTUAL_KEY)),
        use_hyde=bool(eff.get(USE_HYDE_KEY)),
        use_small_to_big=bool(eff.get(USE_SMALL_TO_BIG_KEY)),
        recall_n=int(eff[RECALL_N_KEY]),
        store=rag_ctx.store,
        embedder=rag_ctx.embedder,
    )
    chunks = await retriever.aquery(query, top_k=int(eff[TOP_K_KEY]))
    body = chunks_to_tool_json(chunks)
    return _append_notes(body, notes)
