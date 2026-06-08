from __future__ import annotations

from typing import List

from ..base import BaseRetriever, Chunk
from .vector_retriever import VectorRetriever


class HybridRetriever(BaseRetriever):
    """
    BM25 + dense fusion (planned).

    Until rank-bm25 is wired up, delegates to VectorRetriever so the pipeline
    can use the same constructor slot from FRAMEWORK_DESIGN.md.
    """

    def __init__(
        self,
        vector: VectorRetriever,
        *,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self._vector = vector
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._bm25 = None  # future: BM25 index over chunk texts

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        # TODO: fuse BM25 hits with self._vector.aretrieve(...)
        return await self._vector.aretrieve(query, top_k=top_k)

    @classmethod
    def from_embedder_store(
        cls,
        embedder,
        store,
        **kwargs,
    ) -> "HybridRetriever":
        return cls(VectorRetriever(embedder=embedder, store=store), **kwargs)
