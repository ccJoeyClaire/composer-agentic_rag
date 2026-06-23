"""RAG deployment binding: shared store/embedder, profile defaults, variant cache.

Run (from repo root):
  python -c "from rag.context import bind_rag_context; bind_rag_context(collection='demo', in_memory=True)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rag.build import (
    _make_embedder,
    _make_store,
    build_RAG_indexer,
    build_RAG_retriever,
)
from rag.config import DEFAULT_PROFILE_ID, get_rag_config
from rag.core import RAGIndexer, RAGRetriever
from rag.embedder.openai_embedder import OpenAIEmbedder
from rag.profile_schema import (
    RECALL_N_KEY,
    TOP_K_KEY,
    USE_CONTEXTUAL_KEY,
    USE_HYDE_KEY,
    USE_PREDICT_QUESTIONS_KEY,
    USE_RERANKER_KEY,
    USE_SMALL_TO_BIG_KEY,
    USE_TOKEN_CHUNKER_KEY,
    RagIndexProfile,
    RagSearchProfile,
    default_index_profile,
    default_search_profile,
    normalize_index_profile,
    normalize_search_profile,
)
from rag.store.qdrant_store import QdrantVectorStore

_RetrieverKey = Tuple[bool, bool, bool, bool, int]
_IndexerKey = Tuple[bool, bool, bool, bool]

_SEARCH_OPTION_LABELS = (
    USE_CONTEXTUAL_KEY,
    USE_SMALL_TO_BIG_KEY,
    USE_HYDE_KEY,
    USE_RERANKER_KEY,
)


@dataclass
class RagToolContext:
    """Runtime binding for RAG tools and agent bootstrap.

    Holds shared ``store`` + ``embedder``, per-deployment defaults (from a yaml
    profile id), allow-range gates, and lazy caches of built variants. Pipeline
    bools are chosen per tool call — index and search need not align.

    - Full (``bind_rag_context``): ``store`` + ``embedder`` present; variants
      built and cached on demand from each call's profile.
    - Legacy (``bind_indexer`` / ``bind_retriever``): a prebuilt object is pinned;
      option requests yield explanatory notes instead of rebuilding.
    """

    collection: str = "rag"
    in_memory: bool = False
    store: Optional[QdrantVectorStore] = None
    embedder: Optional[OpenAIEmbedder] = None

    default_search_profile: RagSearchProfile = field(
        default_factory=default_search_profile
    )
    default_index_profile: RagIndexProfile = field(default_factory=default_index_profile)

    allow_token_chunker: bool = True
    allow_contextual: bool = True
    allow_small_to_big: bool = True
    allow_predict_questions: bool = True
    allow_hyde: bool = True
    allow_reranker: bool = True
    max_recall_n: int = field(
        default_factory=lambda: get_rag_config().retriever.recall_n
    )
    max_top_k: int | None = None

    fixed_retriever: Optional[RAGRetriever] = None
    fixed_indexer: Optional[RAGIndexer] = None

    _retriever_cache: Dict[_RetrieverKey, RAGRetriever] = field(default_factory=dict)
    _indexer_cache: Dict[_IndexerKey, RAGIndexer] = field(default_factory=dict)

    @property
    def can_build(self) -> bool:
        return self.store is not None and self.embedder is not None

    def resolve_retriever(
        self,
        profile: RagSearchProfile | None = None,
    ) -> Tuple[Optional[RAGRetriever], List[str], RagSearchProfile]:
        """Return a retriever, notes, and the effective normalized search profile."""
        notes: List[str] = []

        if not self.can_build:
            if self.fixed_retriever is None:
                return None, notes, dict(self.default_search_profile)
            if profile:
                for label in _SEARCH_OPTION_LABELS:
                    if profile.get(label):
                        notes.append(
                            f"{label} requested but this deployment uses a fixed retriever; ran without it."
                        )
                if RECALL_N_KEY in profile:
                    notes.append("recall_n requested but a fixed retriever is bound; ignored.")
            return self.fixed_retriever, notes, dict(self.default_search_profile)

        eff, norm_notes = normalize_search_profile(
            profile,
            defaults=self.default_search_profile,
            allow_contextual=self.allow_contextual,
            allow_small_to_big=self.allow_small_to_big,
            allow_hyde=self.allow_hyde,
            allow_reranker=self.allow_reranker,
            max_recall_n=self.max_recall_n,
            max_top_k=self.max_top_k,
        )
        notes.extend(norm_notes)

        key: _RetrieverKey = (
            bool(eff.get(USE_CONTEXTUAL_KEY)),
            bool(eff.get(USE_SMALL_TO_BIG_KEY)),
            bool(eff.get(USE_HYDE_KEY)),
            bool(eff.get(USE_RERANKER_KEY)),
            int(eff[RECALL_N_KEY]),
        )
        retriever = self._retriever_cache.get(key)
        if retriever is None:
            retriever = build_RAG_retriever(
                self.collection,
                in_memory=self.in_memory,
                use_reranker=bool(eff.get(USE_RERANKER_KEY)),
                use_contextual=bool(eff.get(USE_CONTEXTUAL_KEY)),
                use_hyde=bool(eff.get(USE_HYDE_KEY)),
                use_small_to_big=bool(eff.get(USE_SMALL_TO_BIG_KEY)),
                recall_n=int(eff[RECALL_N_KEY]),
                store=self.store,
                embedder=self.embedder,
            )
            self._retriever_cache[key] = retriever
        return retriever, notes, eff

    def resolve_indexer(
        self,
        profile: RagIndexProfile | None = None,
    ) -> Tuple[Optional[RAGIndexer], List[str], RagIndexProfile]:
        notes: List[str] = []

        if not self.can_build:
            if self.fixed_indexer is None:
                return None, notes, dict(self.default_index_profile)
            if profile:
                for label in (
                    USE_TOKEN_CHUNKER_KEY,
                    USE_CONTEXTUAL_KEY,
                    USE_SMALL_TO_BIG_KEY,
                    USE_PREDICT_QUESTIONS_KEY,
                ):
                    if profile.get(label):
                        notes.append(
                            f"{label} requested but a fixed indexer is bound; ignored."
                        )
            return self.fixed_indexer, notes, dict(self.default_index_profile)

        eff, norm_notes = normalize_index_profile(
            profile,
            defaults=self.default_index_profile,
            allow_token_chunker=self.allow_token_chunker,
            allow_contextual=self.allow_contextual,
            allow_small_to_big=self.allow_small_to_big,
            allow_predict_questions=self.allow_predict_questions,
        )
        notes.extend(norm_notes)

        key: _IndexerKey = (
            bool(eff.get(USE_TOKEN_CHUNKER_KEY)),
            bool(eff.get(USE_CONTEXTUAL_KEY)),
            bool(eff.get(USE_SMALL_TO_BIG_KEY)),
            bool(eff.get(USE_PREDICT_QUESTIONS_KEY)),
        )
        indexer = self._indexer_cache.get(key)
        if indexer is None:
            indexer = build_RAG_indexer(
                self.collection,
                in_memory=self.in_memory,
                use_token_chunker=bool(eff.get(USE_TOKEN_CHUNKER_KEY)),
                use_contextual=bool(eff.get(USE_CONTEXTUAL_KEY)),
                use_predict_questions=bool(eff.get(USE_PREDICT_QUESTIONS_KEY)),
                use_small_to_big=bool(eff.get(USE_SMALL_TO_BIG_KEY)),
                store=self.store,
                embedder=self.embedder,
            )
            self._indexer_cache[key] = indexer
        return indexer, notes, eff


_active_context: RagToolContext = RagToolContext()


def get_active_context() -> RagToolContext:
    """Return the process-wide RAG deployment context (set by ``bind_rag_context``)."""
    return _active_context


def reset_rag_context() -> RagToolContext:
    """Replace the active context with a fresh empty one (tests / isolation)."""
    global _active_context
    _active_context = RagToolContext()
    return _active_context


def bind_rag_context(
    *,
    collection: str,
    in_memory: bool = False,
    profile_id: str = DEFAULT_PROFILE_ID,
    default_top_k: int | None = None,
    default_recall_n: int | None = None,
    max_recall_n: int | None = None,
    max_top_k: int | None = None,
    allow_token_chunker: bool = True,
    allow_contextual: bool = True,
    allow_small_to_big: bool = True,
    allow_predict_questions: bool = True,
    allow_hyde: bool = True,
    allow_reranker: bool = True,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
) -> RagToolContext:
    """Bind shared store/embedder plus deployment defaults and allow-range gates.

    Default search/index bools come from ``arg_config.yaml`` → ``profiles.<id>``
    (default ``baseline``). Each tool call may override any pipeline flag; index
    and search profiles are independent.
    """
    global _active_context

    shared_embedder = _make_embedder(embedder)
    shared_store = _make_store(collection, in_memory=in_memory, store=store)
    retriever_cfg = get_rag_config().retriever
    resolved_max_recall = (
        max_recall_n if max_recall_n is not None else retriever_cfg.recall_n
    )

    search_defaults = default_search_profile(profile_id)
    if default_recall_n is not None:
        search_defaults[RECALL_N_KEY] = default_recall_n
    if default_top_k is not None:
        search_defaults[TOP_K_KEY] = default_top_k

    _active_context = RagToolContext(
        collection=collection,
        in_memory=in_memory,
        store=shared_store,
        embedder=shared_embedder,
        default_search_profile=search_defaults,
        default_index_profile=default_index_profile(profile_id),
        allow_token_chunker=allow_token_chunker,
        allow_contextual=allow_contextual,
        allow_small_to_big=allow_small_to_big,
        allow_predict_questions=allow_predict_questions,
        allow_hyde=allow_hyde,
        allow_reranker=allow_reranker,
        max_recall_n=resolved_max_recall,
        max_top_k=max_top_k,
    )
    return _active_context


def bind_indexer(indexer: RAGIndexer) -> None:
    """Legacy: pin a prebuilt indexer onto the active context (options ignored)."""
    get_active_context().fixed_indexer = indexer


def bind_retriever(retriever: RAGRetriever, *, top_k: int | None = None) -> None:
    """Legacy: pin a prebuilt retriever onto the active context (options ignored)."""
    ctx = get_active_context()
    ctx.fixed_retriever = retriever
    if top_k is not None:
        ctx.default_search_profile = dict(ctx.default_search_profile)
        ctx.default_search_profile[TOP_K_KEY] = top_k
