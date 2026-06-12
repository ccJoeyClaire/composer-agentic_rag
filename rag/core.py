from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, List, Optional

from .base import (
    BaseChunker,
    BaseContextualEnricher,
    BaseEmbedder,
    BaseQueryTransformer,
    BaseReranker,
    BaseRetriever,
    BaseVectorStore,
    Chunk,
    RagContext,
    RagResult,
)
from .document_augmentation.parent_builder import assign_parent_chunks
from .config import get_rag_config


class RAGIndexer:
    """
    Indexing flow: text → chunk → (optional enrich) → embed → store.

    Used offline or on document ingest; no retriever / reranker / HyDE.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        *,
        contextual_enricher: Optional[BaseContextualEnricher] = None,
        predict_question_enricher: Optional[BaseContextualEnricher] = None,
        small_to_big_parent_tokens: Optional[int] = None,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.contextual_enricher = contextual_enricher
        self.predict_question_enricher = predict_question_enricher
        self.small_to_big_parent_tokens = small_to_big_parent_tokens

    async def aindex(self, text: str, source: str = "") -> bool:
        chunks = self.chunker.run(text)
        if source:
            for c in chunks:
                c.metadata.setdefault("source", source)

        if self.small_to_big_parent_tokens:
            chunks = assign_parent_chunks(
                chunks,
                parent_token_budget=self.small_to_big_parent_tokens,
            )

        if self.contextual_enricher:
            chunks = await self.contextual_enricher.aenrich_for_index(
                chunks, source=source
            )

        if self.predict_question_enricher:
            chunks = await self.predict_question_enricher.aenrich_for_index(
                chunks, source=source
            )

        embed_inputs = [
            (c.metadata or {}).get("embed_text", c.content) for c in chunks
        ]
        embeddings = await self.embedder.aembed_texts(embed_inputs)
        await self.store.aadd_chunks(chunks, embeddings)
        return await self.averify_index(source=source, expected_count=len(chunks))

    async def averify_index(
        self,
        source: str,
        *,
        expected_count: Optional[int] = None,
    ) -> bool:
        """
        Return True if the vector store contains indexed chunks for ``source``.

        Pass ``expected_count`` to require an exact chunk count match.
        """
        if not source:
            return False

        stored_count = await self.store.acount_by_source(source)
        if stored_count <= 0:
            return False
        if expected_count is not None and stored_count != expected_count:
            return False
        return True

    def as_tool(self) -> Callable[[str, str], str]:
        def rag_index(text: str, source: str) -> str:
            ok = asyncio.run(self.aindex(text, source=source))
            if ok:
                return f"Successfully indexed document '{source}'."
            return f"Failed to index or verify document '{source}'."

        rag_index.__doc__ = "Index document text into the knowledge base."
        return rag_index


class RAGRetriever:
    """
    Query flow: query → (transform) → retrieve → (contextual chunks) → rerank.

    Only needs a ``BaseRetriever`` (which typically wraps embedder + store).
    No chunker.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        reranker: Optional[BaseReranker] = None,
        query_transformer: Optional[BaseQueryTransformer] = None,
        contextual_enricher: Optional[BaseContextualEnricher] = None,
        recall_n: int | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.contextual_enricher = contextual_enricher
        resolved = recall_n if recall_n is not None else get_rag_config().retriever.recall_n
        self.recall_n = max(1, resolved)

    async def aquery(self, query: str, top_k: int | None = None) -> List[Chunk]:
        effective_top_k = top_k if top_k is not None else get_rag_config().retriever.top_k
        ctx = RagContext(query=query, top_k=effective_top_k)
        return await self._run_query(ctx)

    async def _run_query(self, ctx: RagContext) -> List[Chunk]:
        fetch_k = max(self.recall_n, ctx.top_k) if self.reranker else ctx.top_k

        q = ctx.effective_query
        if self.query_transformer:
            transformed = await self.query_transformer.atransform(q)
            if isinstance(transformed, list):
                q = transformed[0] if transformed else q
            else:
                q = transformed
            hyde_doc = getattr(self.query_transformer, "last_document", None)
            if hyde_doc:
                ctx.metadata["hyde_document"] = hyde_doc
        ctx.working_query = q

        chunks = await self.retriever.aretrieve(ctx.effective_query, top_k=fetch_k)

        if self.contextual_enricher:
            chunks = await self.contextual_enricher.aenrich_chunks(chunks)

        if self.reranker:
            chunks = await self.reranker.arerank(ctx.query, chunks)

        chunks = chunks[: ctx.top_k]
        ctx.chunks = chunks
        return chunks

    async def aquery_result(self, query: str, top_k: int | None = None) -> RagResult:
        chunks = await self.aquery(query, top_k=top_k)
        return RagResult(query=query, chunks=chunks)

    async def aquery_stream(
        self, query: str, top_k: int | None = None
    ) -> AsyncIterator[Chunk]:
        for chunk in await self.aquery(query, top_k=top_k):
            yield chunk

    def as_tool(self, top_k: int | None = None) -> Callable[[str], str]:
        def rag_search(query: str) -> str:
            chunks = asyncio.run(self.aquery(query, top_k=top_k))
            return "\n\n---\n\n".join(c.content for c in chunks)

        rag_search.__doc__ = "Search the knowledge base and return relevant context."
        return rag_search
