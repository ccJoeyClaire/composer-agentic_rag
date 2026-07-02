"""RAG deployment context: shared store, embedder, and collection name only."""

from __future__ import annotations

from dataclasses import dataclass

from rag.build import _make_embedder, _make_store
from rag.config import (
    DEFAULT_INDEX_PROFILE_ID,
    DEFAULT_RETRIEVE_PROFILE_ID,
)
from rag.embedder.openai_embedder import OpenAIEmbedder
from rag.profile_schema import (
    RagIndexProfile,
    RagSearchProfile,
    default_index_profile,
    default_search_profile,
)
from rag.store.qdrant_store import QdrantVectorStore
from tools.context import ToolContextBundle

RAG_CONTEXT_KEY = "rag"
RAG_SEARCH_PROFILE_KEY = "rag_search_profile"
RAG_INDEX_PROFILE_KEY = "rag_index_profile"


@dataclass
class RagToolContext:
    """Shared connections for one RAG deployment (no profile or pipeline cache)."""

    collection: str
    in_memory: bool = False
    store: QdrantVectorStore | None = None
    embedder: OpenAIEmbedder | None = None


async def aclose_rag(ctx: RagToolContext) -> None:
    """Close the shared store."""
    if ctx.store is not None:
        await ctx.store.aclose()
        ctx.store = None
    ctx.embedder = None


def build_rag_context(
    *,
    collection: str,
    in_memory: bool = False,
    store: QdrantVectorStore | None = None,
    embedder: OpenAIEmbedder | None = None,
) -> RagToolContext:
    """Build connection bindings only (no profile defaults)."""
    return RagToolContext(
        collection=collection,
        in_memory=in_memory,
        store=_make_store(collection, in_memory=in_memory, store=store),
        embedder=_make_embedder(embedder),
    )


def bind_rag(
    bundle: ToolContextBundle,
    *,
    collection: str,
    in_memory: bool = False,
    store: QdrantVectorStore | None = None,
    embedder: OpenAIEmbedder | None = None,
    index_profile_id: str = DEFAULT_INDEX_PROFILE_ID,
    retrieve_profile_id: str = DEFAULT_RETRIEVE_PROFILE_ID,
) -> RagToolContext:
    """Register connection context plus optional yaml profile defaults on *bundle*."""
    ctx = build_rag_context(
        collection=collection,
        in_memory=in_memory,
        store=store,
        embedder=embedder,
    )
    bundle.bind(RAG_CONTEXT_KEY, ctx, aclose=lambda: aclose_rag(ctx))
    bundle.bind(RAG_SEARCH_PROFILE_KEY, default_search_profile(retrieve_profile_id))
    bundle.bind(RAG_INDEX_PROFILE_KEY, default_index_profile(index_profile_id))
    return ctx
