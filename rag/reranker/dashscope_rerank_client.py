"""DashScope / Model Studio text-rerank HTTP client.

``qwen3-rerank`` uses the OpenAI-compatible ``/compatible-api/v1/reranks`` endpoint.
``gte-rerank-v2`` and ``qwen3-vl-rerank`` use the legacy DashScope ``text-rerank`` API.

Run (from repo root, requires RERANK_API_KEY or DASHSCOPE_API_KEY):
  python -m rag.reranker.dashscope_rerank_client
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import TypedDict

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_RERANK_MODEL = "qwen3-rerank"
COMPATIBLE_RERANK_MODEL = "qwen3-rerank"
DEFAULT_COMPATIBLE_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
)
DEFAULT_LEGACY_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
DEFAULT_DASHSCOPE_MAX_CONCURRENCY = 2
_RERANK_RETRY_ATTEMPTS = 6
_RERANK_RETRY_BASE_DELAY_S = 1.0
_LEGACY_RERANK_PATH_SUFFIX = "/services/rerank/text-rerank/text-rerank"

_rerank_semaphore: asyncio.Semaphore | None = None
_rerank_semaphore_limit: int | None = None


class RerankDocument(TypedDict):
    text: str


class RerankResultItem(TypedDict, total=False):
    index: int
    relevance_score: float
    document: RerankDocument | str


class RerankApiResponse(TypedDict, total=False):
    status_code: int
    code: str
    message: str
    results: list[RerankResultItem]
    output: dict[str, object]
    request_id: str


@dataclass(frozen=True)
class RerankResult:
    """One ranked document from the rerank API."""

    index: int
    relevance_score: float
    document_text: str | None = None


def uses_compatible_rerank_api(model: str) -> bool:
    """Return True when ``model`` must use ``/compatible-api/v1/reranks``."""
    return model.lower() == COMPATIBLE_RERANK_MODEL


def _is_dashscope_url(url: str | None) -> bool:
    normalized = (url or "").lower()
    return "dashscope" in normalized or "aliyuncs.com" in normalized


def _resolve_endpoint(model: str, base_url: str | None) -> str:
    """Pick the rerank HTTP endpoint for ``model`` and optional ``base_url``."""
    compatible = uses_compatible_rerank_api(model)
    if not base_url:
        return (
            DEFAULT_COMPATIBLE_RERANK_ENDPOINT
            if compatible
            else DEFAULT_LEGACY_RERANK_ENDPOINT
        )

    url = base_url.rstrip("/")
    if url.endswith("/reranks") or url.endswith("text-rerank"):
        return url
    if "/services/rerank/" in url:
        return url

    if compatible:
        if "/compatible-api/" in url:
            return f"{url}/reranks"
        # Common misconfiguration: legacy /api/v1 base with qwen3-rerank model.
        return DEFAULT_COMPATIBLE_RERANK_ENDPOINT

    if url.endswith("/api/v1"):
        return f"{url}{_LEGACY_RERANK_PATH_SUFFIX}"
    return DEFAULT_LEGACY_RERANK_ENDPOINT


def _resolve_api_key(api_key: str | None) -> str:
    resolved = (
        api_key
        or os.environ.get("RERANK_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("EMBEDDING_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    if not resolved:
        raise ValueError(
            "RERANK_API_KEY, DASHSCOPE_API_KEY, EMBEDDING_API_KEY, or LLM_API_KEY is required"
        )
    return resolved


def _resolve_max_concurrency(max_concurrency: int | None, base_url: str | None) -> int | None:
    if max_concurrency is not None:
        return max_concurrency if max_concurrency > 0 else None

    env_val = os.environ.get("RERANK_MAX_CONCURRENCY")
    if env_val is not None:
        limit = int(env_val)
        return limit if limit > 0 else None

    if _is_dashscope_url(base_url):
        return DEFAULT_DASHSCOPE_MAX_CONCURRENCY

    return None


def _semaphore_for(limit: int | None) -> asyncio.Semaphore | None:
    global _rerank_semaphore, _rerank_semaphore_limit
    if limit is None:
        return None
    if _rerank_semaphore is None or _rerank_semaphore_limit != limit:
        _rerank_semaphore = asyncio.Semaphore(limit)
        _rerank_semaphore_limit = limit
    return _rerank_semaphore


def _document_text(document: RerankDocument | str | None) -> str | None:
    if document is None:
        return None
    if isinstance(document, str):
        return document
    text = document.get("text")
    return str(text) if text is not None else None


def _extract_result_items(payload: RerankApiResponse) -> list[RerankResultItem]:
    output = payload.get("output")
    if isinstance(output, dict):
        raw = output.get("results")
        if isinstance(raw, list):
            return raw
    raw = payload.get("results")
    if isinstance(raw, list):
        return raw
    return []


def _parse_results(payload: RerankApiResponse) -> list[RerankResult]:
    parsed: list[RerankResult] = []
    for item in _extract_result_items(payload):
        index = item.get("index")
        score = item.get("relevance_score")
        if index is None or score is None:
            continue
        parsed.append(
            RerankResult(
                index=int(index),
                relevance_score=float(score),
                document_text=_document_text(item.get("document")),
            )
        )
    return parsed


def _raise_api_error(payload: RerankApiResponse, *, http_status: int) -> None:
    code = str(payload.get("code") or "")
    message = str(payload.get("message") or payload)
    if code and code not in {"", "200"}:
        raise RuntimeError(f"DashScope rerank failed ({code}): {message}")
    status_code = payload.get("status_code")
    if status_code is not None and int(status_code) >= 400:
        raise RuntimeError(f"DashScope rerank failed (status_code={status_code}): {message}")
    if http_status >= 400:
        raise RuntimeError(f"DashScope rerank HTTP {http_status}: {message}")


def _build_request_body(
    model: str,
    query: str,
    documents: list[str],
    *,
    top_n: int | None,
    return_documents: bool,
) -> dict[str, object]:
    if uses_compatible_rerank_api(model):
        body: dict[str, object] = {
            "model": model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            body["top_n"] = max(1, top_n)
        return body

    parameters: dict[str, object] = {"return_documents": return_documents}
    if top_n is not None:
        parameters["top_n"] = max(1, top_n)
    return {
        "model": model,
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": parameters,
    }


class DashScopeRerankClient:
    """Async client for DashScope text rerank APIs."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrency: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.model = model or os.environ.get("RERANK_MODEL_ID") or DEFAULT_RERANK_MODEL
        resolved_base = base_url or os.environ.get("RERANK_BASE_URL")
        self.endpoint = _resolve_endpoint(self.model, resolved_base)
        self.api_key = _resolve_api_key(api_key)
        self.timeout_s = timeout_s or float(os.environ.get("RERANK_TIMEOUT_S", "60"))
        self._semaphore = _semaphore_for(
            _resolve_max_concurrency(max_concurrency, resolved_base)
        )

    async def arerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_documents: bool = True,
    ) -> list[RerankResult]:
        """Score and rank documents by relevance to ``query``.

        Args:
            query: Search query text.
            documents: Candidate passage texts (same order as input indices).
            top_n: Return only the top N hits; ``None`` returns all, ranked.
            return_documents: Legacy API only; ask the API to echo document text.

        Returns:
            Results sorted by ``relevance_score`` descending.
        """
        if not documents:
            return []
        if not query.strip():
            return [
                RerankResult(index=i, relevance_score=0.0, document_text=doc)
                for i, doc in enumerate(documents)
            ]

        body = _build_request_body(
            self.model,
            query,
            documents,
            top_n=top_n,
            return_documents=return_documents,
        )
        payload = await self._post_json(body)
        return _parse_results(payload)

    async def ascore_documents(self, query: str, documents: list[str]) -> list[float]:
        """Return relevance scores aligned with the input ``documents`` order."""
        if not documents:
            return []
        ranked = await self.arerank(query, documents, top_n=len(documents))
        scores = [0.0] * len(documents)
        for item in ranked:
            if 0 <= item.index < len(scores):
                scores[item.index] = item.relevance_score
        return scores

    async def _post_json(self, body: dict[str, object]) -> RerankApiResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(_RERANK_RETRY_ATTEMPTS):
            try:
                if self._semaphore is not None:
                    async with self._semaphore:
                        return await self._send_request(headers, body)
                return await self._send_request(headers, body)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 or attempt >= _RERANK_RETRY_ATTEMPTS - 1:
                    raise
                delay = _RERANK_RETRY_BASE_DELAY_S * (2**attempt) + random.uniform(0.0, 0.5)
                await asyncio.sleep(delay)
        raise RuntimeError("DashScope rerank retries exhausted")

    async def _send_request(
        self,
        headers: dict[str, str],
        body: dict[str, object],
    ) -> RerankApiResponse:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(self.endpoint, headers=headers, json=body)
            payload: RerankApiResponse = response.json()
            _raise_api_error(payload, http_status=response.status_code)
            response.raise_for_status()
            return payload


async def _demo_main() -> None:
    import sys

    try:
        rerank_client = DashScopeRerankClient()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    query = "什么是文本排序模型"
    documents = [
        "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
        "量子计算是计算科学的一个前沿领域",
        "预训练语言模型的发展给文本排序模型带来了新的进展",
    ]
    results = await rerank_client.arerank(query, documents, top_n=len(documents))
    print(f"model={rerank_client.model}")
    print(f"endpoint={rerank_client.endpoint}")
    for item in results:
        preview = (item.document_text or documents[item.index])[:60]
        print(f"  [{item.index}] score={item.relevance_score:.4f} | {preview}...")


if __name__ == "__main__":
    asyncio.run(_demo_main())
