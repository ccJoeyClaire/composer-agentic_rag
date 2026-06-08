from __future__ import annotations

from typing import Annotated, Optional

from pydantic import Field

from .core import RAGIndexer, RAGRetriever
from .embedder.openai_embedder import OpenAIEmbedder
from .store.qdrant_store import QdrantVectorStore


def _make_store(
    collection: str,
    *,
    in_memory: bool,
    store: Optional[QdrantVectorStore] = None,
) -> QdrantVectorStore:
    if store is not None:
        return store
    if in_memory:
        return QdrantVectorStore(collection=collection, url=":memory:")
    return QdrantVectorStore(collection=collection, host="127.0.0.1", port=6333)


def _make_embedder(embedder: Optional[OpenAIEmbedder] = None) -> OpenAIEmbedder:
    return embedder if embedder is not None else OpenAIEmbedder()


def build_RAG_indexer(
    collection: Annotated[str, Field(description="Qdrant 集合名")],
    *,
    in_memory: Annotated[
        bool, Field(description="True 时使用 Qdrant :memory:，无需 Docker")
    ] = False,
    use_contextual: Annotated[
        bool, Field(description="是否启用 ContextualEnricher（索引 header + embed 增强）")
    ] = False,
    use_predict_questions: Annotated[
        bool, Field(description="是否启用 PredictQuestionEnricher（LLM 为每个 chunk 生成预设问题）")
    ] = False,
    use_small_to_big: Annotated[
        bool, Field(description="是否小块索引（256 token small / 1536 parent window）")
    ] = False,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
) -> RAGIndexer:
    """组装离线建库用的 :class:`RAGIndexer`（chunk → embed → store）。"""
    from .chunker.semantic_chunker import SemanticChunker
    from .document_augmentation.context_enricher import ContextualEnricher
    from .document_augmentation.predict_question import PredictQuestionEnricher

    embedder = _make_embedder(embedder)
    store = _make_store(collection, in_memory=in_memory, store=store)

    small_tokens = 256 if use_small_to_big else 512
    overlap = 32 if use_small_to_big else 64
    contextual_enricher = ContextualEnricher() if use_contextual else None
    predict_question_enricher = (
        PredictQuestionEnricher() if use_predict_questions else None
    )

    return RAGIndexer(
        chunker=SemanticChunker(chunk_tokens=small_tokens, overlap_tokens=overlap),
        embedder=embedder,
        store=store,
        contextual_enricher=contextual_enricher,
        predict_question_enricher=predict_question_enricher,
        small_to_big_parent_tokens=small_tokens * 6 if use_small_to_big else None,
    )


def build_RAG_retriever(
    collection: Annotated[str, Field(description="Qdrant 集合名，须与 indexer 一致")],
    *,
    in_memory: Annotated[
        bool, Field(description="True 时使用 Qdrant :memory:，无需 Docker")
    ] = False,
    use_reranker: Annotated[
        bool, Field(description="是否挂载 CrossEncoder 精排")
    ] = False,
    use_contextual: Annotated[
        bool, Field(description="是否启用 ContextualEnricher（查询拼接上下文）")
    ] = False,
    use_hyde: Annotated[
        bool, Field(description="是否用 HyDE 改写查询向量")
    ] = False,
    use_small_to_big: Annotated[
        bool, Field(description="是否小块索引、大块返回（256/1536 token）")
    ] = False,
    recall_n: Annotated[
        int, Field(description="向量召回条数（rerank 前）", ge=1)
    ] = 50,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
) -> RAGRetriever:
    """组装查询用的 :class:`RAGRetriever`（transform → retrieve → rerank）。"""
    from .document_augmentation.context_enricher import ContextualEnricher
    from .query_transformer.hyde import HyDETransformer
    from .reranker.cross_encoder_reranker import CrossEncoderReranker
    from .retriever.small_to_big_retriever import SmallToBigRetriever
    from .retriever.vector_retriever import VectorRetriever

    embedder = _make_embedder(embedder)
    store = _make_store(collection, in_memory=in_memory, store=store)

    inner = VectorRetriever(embedder=embedder, store=store)
    retriever = (
        SmallToBigRetriever(
            inner,
            store=store,
            parent_token_budget=1536,
        )
        if use_small_to_big
        else inner
    )

    return RAGRetriever(
        retriever=retriever,
        reranker=CrossEncoderReranker() if use_reranker else None,
        query_transformer=HyDETransformer() if use_hyde else None,
        contextual_enricher=ContextualEnricher() if use_contextual else None,
        recall_n=recall_n,
    )
