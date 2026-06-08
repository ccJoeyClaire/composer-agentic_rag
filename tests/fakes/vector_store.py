"""Test-only vector store (does not modify production rag/ modules)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from rag.base import BaseVectorStore, Chunk


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore(BaseVectorStore):
    """
    Lightweight vector store for integration tests.

    Uses cosine similarity over stored embeddings. Supports aretrieve_by_ids
    when chunk metadata contains ``chunk_id``.
    """

    def __init__(self):
        self._records: List[Tuple[Chunk, List[float]]] = []

    async def aadd_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        for chunk, vector in zip(chunks, embeddings):
            self._records.append((chunk, list(vector)))

    async def asearch(self, query_vector: List[float], top_k: int) -> List[Chunk]:
        scored = []
        for chunk, vector in self._records:
            score = _cosine(query_vector, vector)
            scored.append(
                Chunk(
                    content=chunk.content,
                    metadata=dict(chunk.metadata or {}),
                    score=score,
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    async def acount_by_source(self, source: str) -> int:
        if not source:
            return 0
        return sum(
            1
            for chunk, _ in self._records
            if (chunk.metadata or {}).get("source") == source
        )

    async def aretrieve_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        by_id: Dict[str, Chunk] = {}
        for chunk, _ in self._records:
            cid = (chunk.metadata or {}).get("chunk_id")
            if cid:
                by_id[cid] = chunk
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def clear(self) -> None:
        self._records.clear()
