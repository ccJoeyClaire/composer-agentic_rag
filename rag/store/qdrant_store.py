"""
Qdrant vector store for RAG chunks.

Mental model (qdrant-client):
  - Collection  : named index; vector dimension must be fixed at creation time.
  - Point       : id + vector + payload (JSON dict — store Chunk fields here).
  - upsert      : add or update points (use this from aadd_chunks).
  - query_points: nearest-neighbor search (use this from asearch).

Do not use upload_collection for normal indexing — it is for bulk migration.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from qdrant_client import AsyncQdrantClient, models

from ..base import BaseVectorStore, Chunk
from ..document_augmentation.parent_builder import CHUNK_ID_KEY

# Payload keys — keep stable so search can rebuild Chunk objects.
_PAYLOAD_CONTENT = "content"
_PAYLOAD_METADATA = "metadata"


# 以下是两个转换接口
def _chunk_to_payload(chunk: Chunk) -> dict:
    return {
        _PAYLOAD_CONTENT: chunk.content,
        _PAYLOAD_METADATA: dict(chunk.metadata),
    }


def _payload_to_chunk(payload: dict, *, score: float = 0.0) -> Chunk:
    return Chunk(
        content=payload.get(_PAYLOAD_CONTENT, ""),
        metadata=payload.get(_PAYLOAD_METADATA) or {},
        score=score,
    )


def _hit_to_chunk(hit: models.ScoredPoint) -> Chunk:
    return _payload_to_chunk(hit.payload or {}, score=float(hit.score or 0.0))


def _point_id_for_chunk(chunk: Chunk) -> str:
    chunk_id = (chunk.metadata or {}).get(CHUNK_ID_KEY)
    if chunk_id:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
    return str(uuid.uuid4())


def _point_ids_for_chunk_ids(chunk_ids: List[str]) -> List[str]:
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, cid)) for cid in chunk_ids]


class QdrantVectorStore(BaseVectorStore):
    """
    Async Qdrant backend implementing BaseVectorStore.

    Example (local Docker: docker run -p 6333:6333 qdrant/qdrant):
        store = QdrantVectorStore(collection="my_docs", vector_size=1536)
        await store.ensure_collection()
        await store.aadd_chunks(chunks, embeddings)
        hits = await store.asearch(query_vector, top_k=5)
    """

    def __init__(
        self,
        collection: str,
        *,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: models.Distance = models.Distance.COSINE,
    ):
        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance

        if path:
            self.client = AsyncQdrantClient(
                path=path,
                api_key=api_key,
                trust_env=False,
                check_compatibility=False,
            )
        elif url:
            self.client = AsyncQdrantClient(
                url=url,
                api_key=api_key,
                # Avoid picking up any implicit/system proxy settings that can
                # break localhost access on some Windows setups.
                trust_env=False,
                # Some Qdrant deployments block version endpoints; skip the check.
                check_compatibility=False,
            )
        else:
            self.client = AsyncQdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                trust_env=False,
                check_compatibility=False,
            )

    async def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        """Create the collection if it does not exist."""
        size = vector_size or self.vector_size
        if size is None:
            raise ValueError(
                "vector_size is required before first use "
                "(pass to __init__ or to ensure_collection / aadd_chunks)"
            )
        self.vector_size = size

        exists = await self.client.collection_exists(self.collection)
        if exists: # collection 存在则不发生任何事情
            return

        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=size, distance=self.distance),
        ) #如果 collection 不存在，创建新 collection

    async def aadd_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        if len(chunks) != len(embeddings): #先检查 chunks 和 embeddings 数量是否相等
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return

        dim = len(embeddings[0]) # embeddings 的维度
        if any(len(e) != dim for e in embeddings): # 检查 embeddings 的维度是否一样
            raise ValueError("all embeddings must have the same dimension")
        if self.vector_size is not None and dim != self.vector_size: # 检查当前库的维度和输入向量的维度是否相等
            raise ValueError(
                f"embedding dim {dim} != configured vector_size {self.vector_size}"
            )

        await self.ensure_collection(vector_size=dim)

        points = [
            models.PointStruct(
                id=_point_id_for_chunk(chunk),
                vector=vector,
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, embeddings)
        ]

        await self.client.upsert(collection_name=self.collection, points=points)

    async def asearch(self, query_vector: List[float], top_k: int) -> List[Chunk]:
        if not query_vector:
            return []

        if self.vector_size is None:
            self.vector_size = len(query_vector)
        elif len(query_vector) != self.vector_size:
            raise ValueError(
                f"query_vector dim {len(query_vector)} != vector_size {self.vector_size}"
            )

        # Collection must exist (created during aadd_chunks). Skip create on search-only flows.
        if not await self.client.collection_exists(self.collection):
            return []

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [_hit_to_chunk(hit) for hit in response.points]

    async def acount_by_source(self, source: str) -> int:
        if not source:
            return 0

        if not await self.client.collection_exists(self.collection):
            return 0

        source_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"{_PAYLOAD_METADATA}.source",
                    match=models.MatchValue(value=source),
                )
            ]
        )

        count = 0
        offset = None
        while True:
            records, offset = await self.client.scroll(
                collection_name=self.collection,
                scroll_filter=source_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            count += len(records)
            if offset is None:
                break
        return count

    async def aretrieve_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        if not chunk_ids:
            return []

        if not await self.client.collection_exists(self.collection):
            return []

        point_ids = _point_ids_for_chunk_ids(chunk_ids)
        records = await self.client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_payload=True,
        )

        by_chunk_id = {
            (r.payload or {}).get(_PAYLOAD_METADATA, {}).get(CHUNK_ID_KEY): _payload_to_chunk(
                r.payload or {}
            )
            for r in records
            if r.payload
        }
        return [by_chunk_id[cid] for cid in chunk_ids if cid in by_chunk_id]

    async def aclose(self) -> None:
        await self.client.close()
