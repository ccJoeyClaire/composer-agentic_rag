from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A retrieved or indexed text span.

    metadata is a loose dict (stored as Qdrant payload). Known keys by pipeline stage:

    Chunker:
        heading_path (str): section breadcrumb, e.g. "Intro > Setup"
        start, end (int): character offsets in source document
        boundary_reason (str): why semantic chunker split here

    Indexing:
        source (str): file path or document id
        chunk_id (str): stable id, e.g. "doc.md::3"
        chunk_index (int): position in source document
        section_id (str): source + heading_path
        anchor_window (dict): {"anchor_id", "member_ids"} lazy parent view
        chunk_role (str): "small" | "parent"
        window_member_count (int): members in materialized parent
        doc_title, doc_keywords: document-level augmentation
        contextual_header (str): situating header for LLM / reranker
        embed_text (str): text actually sent to the embedder

    Retrieve (small-to-big):
        parent_id, parent_content (str): query-time materialized parent
        matched_small_content (str): small hit snippet(s)
        matched_chunk_ids (list[str]): child ids that triggered the parent
    """

    content: str
    metadata: dict
    score: float = 0.0


@dataclass
class RagContext:
    """Mutable state passed through orchestration stages."""

    query: str
    top_k: int = 5
    source: str = ""
    working_query: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        return self.working_query if self.working_query is not None else self.query


@dataclass
class RagResult:
    query: str
    chunks: List[Chunk]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    @abstractmethod
    def run(self, text: str) -> List[Chunk]:
        ...


class BaseEmbedder(ABC):
    @abstractmethod
    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    async def aembed_query(self, query: str) -> List[float]:
        ...


class BaseVectorStore(ABC):
    @abstractmethod
    async def aadd_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        ...

    @abstractmethod
    async def asearch(self, query_vector: List[float], top_k: int) -> List[Chunk]:
        ...

    async def aretrieve_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Fetch chunks by stable chunk_id. Default: not supported."""
        return []

    async def acount_by_source(self, source: str) -> int:
        """Return indexed chunk count for ``source``. Override in concrete stores."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support acount_by_source"
        )


class BaseRetriever(ABC):
    @abstractmethod
    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        ...


class BaseReranker(ABC):
    @abstractmethod
    async def arerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        ...


class BaseQueryTransformer(ABC):
    @abstractmethod
    async def atransform(self, query: str) -> str | List[str]:
        ...


class BaseContextualEnricher(ABC):
    """Add surrounding context to chunks at index and retrieve time."""

    async def aenrich_for_index(
        self, chunks: List[Chunk], *, source: str = ""
    ) -> List[Chunk]:
        """Prepare chunks before embedding. Default: no-op."""
        return chunks

    @abstractmethod
    async def aenrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        ...
