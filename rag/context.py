"""RAG deployment binding: shared store/embedder, profile flags, retriever/indexer cache.

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
from rag.config import get_profile, get_rag_config
from rag.core import RAGIndexer, RAGRetriever
from rag.embedder.openai_embedder import OpenAIEmbedder
from rag.store.qdrant_store import QdrantVectorStore

_RetrieverKey = Tuple[bool, bool, int]  # (use_hyde, use_reranker, recall_n)
_IndexerKey = bool  # use_predict_questions


def _resolve_flag(explicit: bool | None, *, from_profile: bool | None, default: bool) -> bool:
    if explicit is not None:
        return explicit
    if from_profile is not None:
        return from_profile
    return default


@dataclass
class RagToolContext:
    """Runtime binding for RAG tools and agent bootstrap.

    Holds a shared ``store`` + ``embedder``, index-coupled settings
    (``use_small_to_big`` / ``use_contextual``), query-time allow-range gates,
    and lazy caches of built retriever/indexer variants. Two binding modes:

    - Full (``bind_rag_context``): ``store`` + ``embedder`` present; query-time
      variants are built and cached on demand.
    - Legacy (``bind_indexer`` / ``bind_retriever``): a prebuilt object is pinned;
      runtime option requests yield explanatory notes instead of rebuilding.
    """

    collection: str = "rag"
    in_memory: bool = False
    store: Optional[QdrantVectorStore] = None
    embedder: Optional[OpenAIEmbedder] = None

    use_small_to_big: bool = False
    use_contextual: bool = False

    allow_hyde: bool = True
    allow_reranker: bool = True
    default_recall_n: int = field(
        default_factory=lambda: get_rag_config().retriever.recall_n
    )
    max_recall_n: int = field(
        default_factory=lambda: get_rag_config().retriever.recall_n
    )
    default_top_k: int = field(
        default_factory=lambda: get_rag_config().retriever.top_k
    )

    allow_predict_questions: bool = True

    fixed_retriever: Optional[RAGRetriever] = None
    fixed_indexer: Optional[RAGIndexer] = None

    _retriever_cache: Dict[_RetrieverKey, RAGRetriever] = field(default_factory=dict)
    _indexer_cache: Dict[_IndexerKey, RAGIndexer] = field(default_factory=dict)

    @property
    def can_build(self) -> bool:
        return self.store is not None and self.embedder is not None

    def resolve_retriever(
        self,
        *,
        use_hyde: bool,
        use_reranker: bool,
        recall_n: Optional[int],
    ) -> Tuple[Optional[RAGRetriever], List[str]]:
        """Return a retriever for the requested options plus any fallback notes."""
        notes: List[str] = []

        if not self.can_build:
            if self.fixed_retriever is None:
                return None, notes
            for label, requested in (("use_hyde", use_hyde), ("use_reranker", use_reranker)):
                if requested:
                    notes.append(
                        f"{label} requested but this deployment uses a fixed retriever; ran without it."
                    )
            if recall_n is not None:
                notes.append("recall_n requested but a fixed retriever is bound; ignored.")
            return self.fixed_retriever, notes

        eff_hyde = use_hyde
        if use_hyde and not self.allow_hyde:
            eff_hyde = False
            notes.append("use_hyde is disabled in this deployment; ran without it.")

        eff_reranker = use_reranker
        if use_reranker and not self.allow_reranker:
            eff_reranker = False
            notes.append("use_reranker is disabled in this deployment; ran without it.")

        eff_recall = self.default_recall_n if recall_n is None else recall_n
        if eff_recall < 1:
            eff_recall = 1
        if eff_recall > self.max_recall_n:
            notes.append(
                f"recall_n {eff_recall} exceeds max {self.max_recall_n}; clamped."
            )
            eff_recall = self.max_recall_n

        key: _RetrieverKey = (eff_hyde, eff_reranker, eff_recall)
        retriever = self._retriever_cache.get(key)
        if retriever is None:
            retriever = build_RAG_retriever(
                self.collection,
                in_memory=self.in_memory,
                use_reranker=eff_reranker,
                use_contextual=self.use_contextual,
                use_hyde=eff_hyde,
                use_small_to_big=self.use_small_to_big,
                recall_n=eff_recall,
                store=self.store,
                embedder=self.embedder,
            )
            self._retriever_cache[key] = retriever
        return retriever, notes

    def resolve_indexer(
        self,
        *,
        use_predict_questions: bool,
    ) -> Tuple[Optional[RAGIndexer], List[str]]:
        notes: List[str] = []

        if not self.can_build:
            if self.fixed_indexer is None:
                return None, notes
            if use_predict_questions:
                notes.append(
                    "use_predict_questions requested but a fixed indexer is bound; ignored."
                )
            return self.fixed_indexer, notes

        eff_predict = use_predict_questions
        if use_predict_questions and not self.allow_predict_questions:
            eff_predict = False
            notes.append("use_predict_questions is disabled in this deployment; ran without it.")

        indexer = self._indexer_cache.get(eff_predict)
        if indexer is None:
            indexer = build_RAG_indexer(
                self.collection,
                in_memory=self.in_memory,
                use_contextual=self.use_contextual,
                use_predict_questions=eff_predict,
                use_small_to_big=self.use_small_to_big,
                store=self.store,
                embedder=self.embedder,
            )
            self._indexer_cache[eff_predict] = indexer
        return indexer, notes


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
    profile_id: str | None = None,
    use_small_to_big: bool | None = None,
    use_contextual: bool | None = None,
    allow_hyde: bool | None = None,
    allow_reranker: bool | None = None,
    default_top_k: int | None = None,
    default_recall_n: int | None = None,
    max_recall_n: int | None = None,
    allow_predict_questions: bool | None = None,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
) -> RagToolContext:
    """Bind shared store/embedder plus deployment allow-range for RAG tools.

    When ``profile_id`` is set, index-coupled flags and allow-range defaults are
    taken from ``arg_config.yaml`` → ``profiles.<id>`` unless overridden explicitly.

    Query-time options (``use_hyde`` / ``use_reranker`` / ``recall_n`` / ``top_k``)
    remain runtime-selectable within the allow-range; index-coupled settings are
    fixed here and shared by both index and search tools.
    """
    global _active_context

    profile = get_profile(get_rag_config(), profile_id) if profile_id is not None else None
    profile_flags = profile if profile is not None else None

    shared_embedder = _make_embedder(embedder)
    shared_store = _make_store(collection, in_memory=in_memory, store=store)
    retriever_cfg = get_rag_config().retriever
    resolved_top_k = default_top_k if default_top_k is not None else retriever_cfg.top_k
    resolved_recall_n = (
        default_recall_n if default_recall_n is not None else retriever_cfg.recall_n
    )
    resolved_max_recall = (
        max_recall_n if max_recall_n is not None else resolved_recall_n
    )

    _active_context = RagToolContext(
        collection=collection,
        in_memory=in_memory,
        store=shared_store,
        embedder=shared_embedder,
        use_small_to_big=_resolve_flag(
            use_small_to_big,
            from_profile=profile_flags.use_small_to_big if profile_flags else None,
            default=False,
        ),
        use_contextual=_resolve_flag(
            use_contextual,
            from_profile=profile_flags.use_contextual if profile_flags else None,
            default=False,
        ),
        allow_hyde=_resolve_flag(
            allow_hyde,
            from_profile=profile_flags.use_hyde if profile_flags else None,
            default=True,
        ),
        allow_reranker=_resolve_flag(
            allow_reranker,
            from_profile=profile_flags.use_reranker if profile_flags else None,
            default=True,
        ),
        default_recall_n=resolved_recall_n,
        max_recall_n=resolved_max_recall,
        default_top_k=resolved_top_k,
        allow_predict_questions=_resolve_flag(
            allow_predict_questions,
            from_profile=profile_flags.use_predict_questions if profile_flags else None,
            default=True,
        ),
    )
    return _active_context


def bind_indexer(indexer: RAGIndexer) -> None:
    """Legacy: pin a prebuilt indexer onto the active context (options ignored)."""
    get_active_context().fixed_indexer = indexer


def bind_retriever(retriever: RAGRetriever, *, top_k: int | None = None) -> None:
    """Legacy: pin a prebuilt retriever onto the active context (options ignored)."""
    ctx = get_active_context()
    ctx.fixed_retriever = retriever
    ctx.default_top_k = top_k if top_k is not None else get_rag_config().retriever.top_k
