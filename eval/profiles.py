from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from eval.paths import REPO_ROOT
from rag.build import build_RAG_indexer, build_RAG_retriever
from rag.core import RAGIndexer, RAGRetriever
from rag.embedder.openai_embedder import OpenAIEmbedder
from rag.store.qdrant_store import QdrantVectorStore

LOCAL_QDRANT_DIR = REPO_ROOT / "eval" / ".cache" / "local_qdrant"


def index_concurrency() -> int:
    return max(1, int(os.environ.get("EVAL_INDEX_CONCURRENCY", "3")))


def llm_concurrency() -> int:
    return max(1, int(os.environ.get("EVAL_LLM_CONCURRENCY", "15")))


@dataclass(frozen=True)
class RAGProfile:
    profile_id: str
    use_contextual: bool = False
    use_small_to_big: bool = False
    use_predict_questions: bool = False
    use_hyde: bool = False
    use_reranker: bool = False

    @classmethod
    def get(cls, profile_id: str) -> "RAGProfile":
        for profile in SMOKE_PROFILES:
            if profile.profile_id == profile_id:
                return profile
        supported = ", ".join(p.profile_id for p in SMOKE_PROFILES)
        raise ValueError(f"Unknown profile {profile_id!r}. Supported: {supported}")


SMOKE_PROFILES: List[RAGProfile] = [
    RAGProfile("baseline"),
    RAGProfile("contextual", use_contextual=True),
    RAGProfile("s2b", use_small_to_big=True),
    RAGProfile("predict_q", use_predict_questions=True),
    RAGProfile(
        "full",
        use_contextual=True,
        use_small_to_big=True,
        use_hyde=True,
        use_reranker=True,
    ),
]


def collection_name(dataset: str, profile_id: str) -> str:
    return f"eval_{dataset}_{profile_id}"


def build_store(
    collection: str,
    *,
    in_memory: bool,
    host: str = "127.0.0.1",
    port: int = 6333,
) -> QdrantVectorStore:
    if in_memory:
        # Shared local Qdrant dir so index + search see the same collections (:memory: URL unsupported).
        LOCAL_QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        return QdrantVectorStore(collection=collection, path=str(LOCAL_QDRANT_DIR))
    return QdrantVectorStore(collection=collection, host=host, port=port)


def build_indexer_for_profile(
    profile: RAGProfile,
    collection: str,
    *,
    in_memory: bool,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
) -> RAGIndexer:
    predict_concurrency = llm_concurrency() if profile.use_predict_questions else None
    if store is None:
        store = build_store(collection, in_memory=in_memory)
    return build_RAG_indexer(
        collection,
        in_memory=False,
        use_contextual=profile.use_contextual,
        use_predict_questions=profile.use_predict_questions,
        use_small_to_big=profile.use_small_to_big,
        predict_question_max_concurrency=predict_concurrency,
        store=store,
        embedder=embedder,
    )


def build_retriever_for_profile(
    profile: RAGProfile,
    collection: str,
    *,
    in_memory: bool,
    store: Optional[QdrantVectorStore] = None,
    embedder: Optional[OpenAIEmbedder] = None,
    recall_n: int = 50,
    top_k: int = 3,
) -> RAGRetriever:
    if store is None:
        store = build_store(collection, in_memory=in_memory)
    return build_RAG_retriever(
        collection,
        in_memory=False,
        use_reranker=profile.use_reranker,
        use_contextual=profile.use_contextual,
        use_hyde=profile.use_hyde,
        use_small_to_big=profile.use_small_to_big,
        recall_n=recall_n,
        store=store,
        embedder=embedder,
    )
