from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from ..base import BaseReranker, Chunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 32


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder rerank: score (query, chunk) pairs and sort by relevance.

    Default model: ms-marco-MiniLM-L-6-v2 (sentence-transformers).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or os.environ.get(
            "CROSS_ENCODER_MODEL", DEFAULT_MODEL
        )
        env_bs = os.environ.get("CROSS_ENCODER_BATCH_SIZE")
        self.batch_size = batch_size or (int(env_bs) if env_bs else DEFAULT_BATCH_SIZE)
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _rerank_sync(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        model = self._get_model()
        pairs = [(query, c.content) for c in chunks] # query 和 chunk 两两配对
        scores = model.predict(
            pairs,
            batch_size=max(1, self.batch_size),
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]), # 根据 score 重排序
            reverse=True,
        )
        return [
            Chunk(
                content=c.content,
                metadata=dict(c.metadata),
                score=float(s),
            )
            for c, s in ranked
        ]

    async def arerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        asyncio.to_thread(...) 会：
        1. 从默认线程池里拿一个工作线程
        2. 在这个线程里执行 _rerank_sync(query, chunks)
        3. 主协程用 await 等待它完成，再拿到返回值
        """
        if not query.strip() or not chunks:
            return list(chunks) # 避免没有 chunk 或 query 的情况
        return await asyncio.to_thread(self._rerank_sync, query, chunks) # rerank_sync 是主要的执行逻辑

