"""DashScope qwen3-rerank adapter for the RAG rerank stage."""

from __future__ import annotations

from typing import List

from ..base import BaseReranker, Chunk
from .dashscope_rerank_client import DashScopeRerankClient


class DashScopeReranker(BaseReranker):
    """Rerank retrieved chunks via DashScope ``text-rerank`` API."""

    def __init__(self, client: DashScopeRerankClient | None = None) -> None:
        self._client = client or DashScopeRerankClient()

    async def arerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not query.strip() or not chunks:
            return list(chunks)

        documents = [chunk.content for chunk in chunks]
        ranked = await self._client.arerank(
            query,
            documents,
            top_n=len(documents),
            return_documents=False,
        )
        score_by_index = {item.index: item.relevance_score for item in ranked}
        ordered_indices = [item.index for item in ranked]

        reranked: List[Chunk] = []
        for index in ordered_indices:
            chunk = chunks[index]
            reranked.append(
                Chunk(
                    content=chunk.content,
                    metadata=dict(chunk.metadata),
                    score=float(score_by_index.get(index, 0.0)),
                )
            )
        return reranked


async def _demo_main() -> None:
    """Integration smoke: rerank hardcoded passages for one query.

    Run (from repo root):
      python -m rag.reranker.dashscope_reranker
    """
    query = "What is retrieval-augmented generation?"
    chunks = [
        Chunk(content="RAG combines retrieval with language models.", metadata={"chunk_id": "a"}),
        Chunk(content="The weather in Paris is sunny.", metadata={"chunk_id": "b"}),
        Chunk(content="Vector databases store embeddings for search.", metadata={"chunk_id": "c"}),
    ]
    print("Before:")
    for chunk in chunks:
        print(f"  {chunk.metadata.get('chunk_id')}: {chunk.content[:50]}")

    reranker = DashScopeReranker()
    ranked = await reranker.arerank(query, chunks)
    print("\nAfter rerank:")
    for chunk in ranked:
        print(
            f"  {chunk.metadata.get('chunk_id')}: "
            f"score={chunk.score:.4f} | {chunk.content[:50]}"
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo_main())
