from __future__ import annotations

from typing import Annotated, Optional

from pydantic import Field

from .base import BaseChunker
from .config import get_rag_config
from .core import RAGIndexer, RAGRetriever
from .embedder.openai_embedder import OpenAIEmbedder
from .store.qdrant_store import QdrantVectorStore

PARENT_WINDOW_MULTIPLIER = 6


def parent_window_tokens(chunk_tokens: int | None = None) -> int:
    tokens = chunk_tokens
    if tokens is None:
        tokens = get_rag_config().chunker.chunk_tokens
    return tokens * PARENT_WINDOW_MULTIPLIER


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


def _make_chunker(
    *,
    use_token_chunker: bool,
    chunk_tokens: int,
    overlap_tokens: int,
) -> BaseChunker:
    if use_token_chunker:
        from .chunker.token_chunker import TokenChunker

        return TokenChunker(
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

    from .chunker.semantic_chunker import SemanticChunker

    chunker_cfg = get_rag_config().chunker
    return SemanticChunker(
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        break_similarity=chunker_cfg.break_similarity,
        min_chunk_tokens=chunker_cfg.min_chunk_tokens,
    )


def build_RAG_indexer(
    collection: Annotated[str, Field(description="Qdrant 集合名")],
    *,
    in_memory: Annotated[
        bool, Field(description="True 时使用 Qdrant :memory:，无需 Docker")
    ] = False,
    use_token_chunker: Annotated[
        bool, Field(description="是否用 TokenChunker 硬切（对照组；默认 SemanticChunker）")
    ] = False,
    use_contextual: Annotated[
        bool, Field(description="是否启用 ContextualEnricher（索引 header + embed 增强）")
    ] = False,
    use_predict_questions: Annotated[
        bool, Field(description="是否启用 PredictQuestionEnricher（LLM 为每个 chunk 生成预设问题）")
    ] = False,
    use_small_to_big: Annotated[
        bool,
        Field(
            description=(
                "是否启用 small-to-big"
                f"（{get_rag_config().chunker.chunk_tokens} child chunk / "
                f"{parent_window_tokens()} parent window）"
            )
        ),
    ] = False,
    predict_question_max_concurrency: Optional[int] = None,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
    chunk_tokens: int | None = None,
    overlap: int | None = None,
) -> RAGIndexer:
    """组装离线建库用的 :class:`RAGIndexer`（chunk → embed → store）。"""
    from .document_augmentation.context_enricher import ContextualEnricher
    from .document_augmentation.predict_question import PredictQuestionEnricher

    chunker_cfg = get_rag_config().chunker
    resolved_chunk_tokens = (
        chunk_tokens if chunk_tokens is not None else chunker_cfg.chunk_tokens
    )
    resolved_overlap = overlap if overlap is not None else chunker_cfg.overlap_tokens

    embedder = _make_embedder(embedder)
    store = _make_store(collection, in_memory=in_memory, store=store)

    contextual_enricher = ContextualEnricher() if use_contextual else None
    predict_kwargs: dict = {}
    if predict_question_max_concurrency is not None:
        predict_kwargs["max_concurrency"] = predict_question_max_concurrency
    predict_question_enricher = (
        PredictQuestionEnricher(**predict_kwargs) if use_predict_questions else None
    )

    return RAGIndexer(
        chunker=_make_chunker(
            use_token_chunker=use_token_chunker,
            chunk_tokens=resolved_chunk_tokens,
            overlap_tokens=resolved_overlap,
        ),
        embedder=embedder,
        store=store,
        contextual_enricher=contextual_enricher,
        predict_question_enricher=predict_question_enricher,
        small_to_big_parent_tokens=(
            parent_window_tokens(resolved_chunk_tokens) if use_small_to_big else None
        ),
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
        bool,
        Field(
            description=(
                "是否启用 small-to-big"
                f"（{get_rag_config().chunker.chunk_tokens} child / "
                f"{parent_window_tokens()} parent）"
            )
        ),
    ] = False,
    recall_n: Annotated[
        int | None, Field(description="向量召回条数（rerank 前）", ge=1)
    ] = None,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
    chunk_tokens: int | None = None,
) -> RAGRetriever:
    """组装查询用的 :class:`RAGRetriever`（transform → retrieve → rerank）。"""
    from .document_augmentation.context_enricher import ContextualEnricher
    from .query_transformer.hyde import HyDETransformer
    from .reranker.cross_encoder_reranker import CrossEncoderReranker
    from .retriever.small_to_big_retriever import SmallToBigRetriever
    from .retriever.vector_retriever import VectorRetriever

    chunker_cfg = get_rag_config().chunker
    retriever_cfg = get_rag_config().retriever
    resolved_chunk_tokens = (
        chunk_tokens if chunk_tokens is not None else chunker_cfg.chunk_tokens
    )
    resolved_recall_n = recall_n if recall_n is not None else retriever_cfg.recall_n

    embedder = _make_embedder(embedder)
    store = _make_store(collection, in_memory=in_memory, store=store)

    inner = VectorRetriever(embedder=embedder, store=store)
    retriever = (
        SmallToBigRetriever(
            inner,
            store=store,
            parent_token_budget=parent_window_tokens(resolved_chunk_tokens),
        )
        if use_small_to_big
        else inner
    )

    return RAGRetriever(
        retriever=retriever,
        reranker=CrossEncoderReranker() if use_reranker else None,
        query_transformer=HyDETransformer() if use_hyde else None,
        contextual_enricher=ContextualEnricher() if use_contextual else None,
        recall_n=resolved_recall_n,
    )
