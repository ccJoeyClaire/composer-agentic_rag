from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Optional, Tuple

from pydantic import Field

from rag.build import (
    _make_embedder,
    _make_store,
    build_RAG_indexer,
    build_RAG_retriever,
)
from rag.core import RAGIndexer, RAGRetriever
from rag.embedder.openai_embedder import OpenAIEmbedder
from rag.store.qdrant_store import QdrantVectorStore
from tools.registry import local_tool

# Cache keys
_RetrieverKey = Tuple[bool, bool, int]  # (use_hyde, use_reranker, recall_n)
_IndexerKey = bool  # use_predict_questions


@dataclass
class RagToolContext:
    """Runtime binding for the RAG tools.

    Holds a shared ``store`` + ``embedder``, the *fixed* index-coupled settings
    (``use_small_to_big`` / ``use_contextual``), the query-time allow-range that
    gates what the LLM may toggle, and lazy caches of built retriever/indexer
    variants. Two binding modes are supported:

    - Full (``bind_rag_context``): ``store`` + ``embedder`` present, so query-time
      variants can be built and cached on demand.
    - Legacy (``bind_indexer`` / ``bind_retriever``): a prebuilt object is pinned;
      options are not buildable and requesting them yields an explanatory note.
    """

    collection: str = "rag"
    in_memory: bool = False
    store: Optional[QdrantVectorStore] = None
    embedder: Optional[OpenAIEmbedder] = None

    # Fixed index-coupled settings (shared by index + search).
    use_small_to_big: bool = False
    use_contextual: bool = False

    # Query-time allow-range + defaults.
    allow_hyde: bool = True
    allow_reranker: bool = True
    default_recall_n: int = 50
    max_recall_n: int = 50
    default_top_k: int = 5

    # Index-time allow-range.
    allow_predict_questions: bool = True

    # Legacy pinned objects (used when store/embedder are absent).
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
            # Index-coupled flags come from the bound context (fixed); only the
            # query-time flags vary per cached variant.
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
            # Index-coupled flags come from the bound context (fixed); only the
            # index-only predict-questions flag varies per cached variant.
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


_context: RagToolContext = RagToolContext()


def bind_rag_context(
    *,
    collection: str,
    in_memory: bool = False,
    use_small_to_big: bool = False,
    use_contextual: bool = False,
    allow_hyde: bool = True,
    allow_reranker: bool = True,
    default_top_k: int = 5,
    default_recall_n: int = 50,
    max_recall_n: int = 50,
    allow_predict_questions: bool = True,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
) -> RagToolContext:
    """Bind a pluggable RAG context: shared store/embedder + allow-range.

    Query-time options (``use_hyde`` / ``use_reranker`` / ``recall_n`` / ``top_k``)
    become runtime-selectable within the allow-range; index-coupled settings
    (``use_small_to_big`` / ``use_contextual``) are fixed here and shared by both
    the index and search tools.
    """
    global _context
    shared_embedder = _make_embedder(embedder)
    shared_store = _make_store(collection, in_memory=in_memory, store=store)
    _context = RagToolContext(
        collection=collection,
        in_memory=in_memory,
        store=shared_store,
        embedder=shared_embedder,
        use_small_to_big=use_small_to_big,
        use_contextual=use_contextual,
        allow_hyde=allow_hyde,
        allow_reranker=allow_reranker,
        default_recall_n=default_recall_n,
        max_recall_n=max_recall_n,
        default_top_k=default_top_k,
        allow_predict_questions=allow_predict_questions,
    )
    return _context


def bind_indexer(indexer: RAGIndexer) -> None:
    """Legacy: pin a prebuilt indexer onto the active context (options ignored)."""
    _context.fixed_indexer = indexer


def bind_retriever(retriever: RAGRetriever, *, top_k: int = 5) -> None:
    """Legacy: pin a prebuilt retriever onto the active context (options ignored)."""
    _context.fixed_retriever = retriever
    _context.default_top_k = top_k


def _run_async(coro):
    """Run coroutine from sync tool entrypoints (works inside asyncio loops too)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

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
    indexer, notes = _context.resolve_indexer(use_predict_questions=use_predict_questions)
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
    retriever, notes = _context.resolve_retriever(
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        recall_n=recall_n,
    )
    if retriever is None:
        return "RAG retriever not bound. Call bind_rag_context() or bind_retriever() at startup."

    effective_top_k = top_k if top_k is not None else _context.default_top_k
    chunks = _run_async(retriever.aquery(query, top_k=effective_top_k))
    body = "\n\n---\n\n".join(c.content for c in chunks)
    return _append_notes(body, notes)
