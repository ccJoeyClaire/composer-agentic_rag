"""Sync embedding client dedicated to semantic chunking.

Kept separate from ``rag.embedder`` so chunking does not depend on the
indexing embedder implementation.
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_EMBEDDING_BATCH_SIZE = 2048
DASHSCOPE_EMBEDDING_BATCH_SIZE = 10


def _resolve_batch_size(
    *,
    batch_size: int | None,
    base_url: str | None,
) -> int:
    if batch_size is not None:
        return max(1, batch_size)

    env_bs = os.environ.get("EMBEDDING_BATCH_SIZE")
    if env_bs:
        return max(1, int(env_bs))

    url = (base_url or "").lower()
    if "dashscope" in url or "aliyuncs.com" in url:
        return DASHSCOPE_EMBEDDING_BATCH_SIZE

    return DEFAULT_EMBEDDING_BATCH_SIZE


class ChunkerEmbeddingClient:
    """OpenAI-compatible sync client for paragraph embeddings during chunking."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model = (
            model or os.environ.get("EMBEDDING_MODEL_ID") or "text-embedding-3-small"
        )
        api_key = api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get(
            "LLM_API_KEY"
        )
        base_url = base_url or os.environ.get("EMBEDDING_BASE_URL") or os.environ.get(
            "LLM_BASE_URL"
        )
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY or LLM_API_KEY is required")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.batch_size = _resolve_batch_size(batch_size=batch_size, base_url=base_url)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of paragraph texts (sync, batched for provider limits).

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            One embedding vector per input text, in the same order.
        """
        if not texts:
            return []

        out: List[List[float]] = []
        bs = max(1, self.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            response = self._client.embeddings.create(
                input=batch,
                model=self.model,
            )
            out.extend(item.embedding for item in response.data)
        return out
