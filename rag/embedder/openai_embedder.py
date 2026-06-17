import os
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ..base import BaseEmbedder

load_dotenv()

# OpenAI official supports large batches; DashScope compatible-mode caps at 10.
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


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
    ):
        self.model = model or os.environ.get("EMBEDDING_MODEL_ID") or "text-embedding-3-small"
        api_key = api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("LLM_API_KEY")
        base_url = base_url or os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("LLM_BASE_URL")
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY or LLM_API_KEY is required")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.batch_size = _resolve_batch_size(batch_size=batch_size, base_url=base_url)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        out: List[List[float]] = []
        bs = max(1, self.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            out.extend(item.embedding for item in response.data)
        return out

    async def aembed_query(self, query: str) -> List[float]:
        vectors = await self.aembed_texts([query])
        return vectors[0]


async def _demo_main() -> None:
    """Integration smoke: embed query + batch and print vector dimension.

    Run (from repo root):
      python -m rag.embedder.openai_embedder
    """
    import asyncio
    import sys

    try:
        embedder = OpenAIEmbedder()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    query_vec = await embedder.aembed_query("What is RAG?")
    batch = await embedder.aembed_texts(["Paris is in France.", "Berlin is in Germany."])
    print(f"model={embedder.model}")
    print(f"query dim={len(query_vec)}")
    print(f"batch dims={[len(v) for v in batch]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo_main())
