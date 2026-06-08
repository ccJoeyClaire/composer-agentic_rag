from __future__ import annotations

from typing import List

from ..base import BaseEmbedder, BaseRetriever, BaseVectorStore, Chunk


class VectorRetriever(BaseRetriever):
    """
    Dense retrieval: embed the query, then search the vector store.

        retriever = VectorRetriever(embedder=embedder, store=store)
        chunks = await retriever.aretrieve("what is RAG?", top_k=5)
    """

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore):
        self.embedder = embedder
        self.store = store

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        if not query.strip():
            return []

        query_vector = await self.embedder.aembed_query(query) #将 query 转换成向量后查询，注意embedding模型要保持一致
        return await self.store.asearch(query_vector, top_k=top_k)
