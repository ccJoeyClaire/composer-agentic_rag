"""Unit tests for DashScope rerank client and adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rag.base import Chunk
from rag.reranker.dashscope_rerank_client import (
    DEFAULT_COMPATIBLE_RERANK_ENDPOINT,
    DEFAULT_LEGACY_RERANK_ENDPOINT,
    DashScopeRerankClient,
    _parse_results,
    _resolve_endpoint,
)
from rag.reranker.dashscope_reranker import DashScopeReranker
from rag.reranker.factory import make_reranker, resolve_rerank_backend

pytestmark = pytest.mark.unit


def test_resolve_endpoint_qwen3_rerank_uses_compatible_api() -> None:
    assert _resolve_endpoint("qwen3-rerank", None) == DEFAULT_COMPATIBLE_RERANK_ENDPOINT


def test_resolve_endpoint_qwen3_rerank_ignores_legacy_base_url() -> None:
    assert _resolve_endpoint(
        "qwen3-rerank",
        "https://dashscope.aliyuncs.com/api/v1",
    ) == DEFAULT_COMPATIBLE_RERANK_ENDPOINT


def test_resolve_endpoint_qwen3_rerank_appends_reranks_to_compatible_base() -> None:
    assert _resolve_endpoint(
        "qwen3-rerank",
        "https://dashscope.aliyuncs.com/compatible-api/v1",
    ) == DEFAULT_COMPATIBLE_RERANK_ENDPOINT


def test_resolve_endpoint_legacy_model_appends_service_path() -> None:
    assert _resolve_endpoint(
        "gte-rerank-v2",
        "https://dashscope.aliyuncs.com/api/v1",
    ) == DEFAULT_LEGACY_RERANK_ENDPOINT


def test_resolve_endpoint_keeps_full_rerank_url() -> None:
    url = "https://example.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    assert _resolve_endpoint("gte-rerank-v2", url) == url


def test_parse_results_qwen3_rerank_shape() -> None:
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.9, "document": {"text": "a"}},
            {"index": 2, "relevance_score": 0.3, "document": {"text": "c"}},
        ]
    }
    parsed = _parse_results(payload)
    assert len(parsed) == 2
    assert parsed[0].index == 0
    assert parsed[0].relevance_score == pytest.approx(0.9)


def test_parse_results_qwen3_vl_shape() -> None:
    payload = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.7, "document": {"text": "b"}},
            ]
        }
    }
    parsed = _parse_results(payload)
    assert len(parsed) == 1
    assert parsed[0].index == 1


@pytest.mark.asyncio
async def test_client_qwen3_rerank_posts_flat_body() -> None:
    client = DashScopeRerankClient(
        model="qwen3-rerank",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        max_concurrency=None,
    )
    mock_response = httpx.Response(
        200,
        json={
            "results": [
                {"index": 0, "relevance_score": 0.93},
                {"index": 2, "relevance_score": 0.34},
                {"index": 1, "relevance_score": 0.05},
            ],
        },
        request=httpx.Request("POST", client.endpoint),
    )

    with patch(
        "rag.reranker.dashscope_rerank_client.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as post_mock:
        results = await client.arerank(
            "query",
            ["doc-a", "doc-b", "doc-c"],
            top_n=3,
        )

    assert client.endpoint == DEFAULT_COMPATIBLE_RERANK_ENDPOINT
    assert post_mock.await_count == 1
    body = post_mock.await_args.kwargs["json"]
    assert body == {
        "model": "qwen3-rerank",
        "query": "query",
        "documents": ["doc-a", "doc-b", "doc-c"],
        "top_n": 3,
    }
    assert results[0].index == 0
    assert results[0].relevance_score == pytest.approx(0.93)


@pytest.mark.asyncio
async def test_client_legacy_model_posts_nested_body() -> None:
    client = DashScopeRerankClient(
        model="gte-rerank-v2",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        max_concurrency=None,
    )
    mock_response = httpx.Response(
        200,
        json={
            "output": {
                "results": [
                    {"index": 0, "relevance_score": 0.8},
                ]
            }
        },
        request=httpx.Request("POST", client.endpoint),
    )

    with patch(
        "rag.reranker.dashscope_rerank_client.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as post_mock:
        await client.arerank("query", ["doc-a"], top_n=1)

    body = post_mock.await_args.kwargs["json"]
    assert body["model"] == "gte-rerank-v2"
    assert body["input"]["query"] == "query"
    assert body["input"]["documents"] == ["doc-a"]
    assert body["parameters"]["top_n"] == 1


@pytest.mark.asyncio
async def test_client_ascore_documents_aligns_with_input_order() -> None:
    client = DashScopeRerankClient(
        model="qwen3-rerank",
        api_key="test-key",
        max_concurrency=None,
    )
    mock_response = httpx.Response(
        200,
        json={
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.4},
                {"index": 1, "relevance_score": 0.1},
            ],
        },
        request=httpx.Request("POST", client.endpoint),
    )

    with patch(
        "rag.reranker.dashscope_rerank_client.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        scores = await client.ascore_documents("query", ["a", "b", "c"])

    assert scores == [pytest.approx(0.9), pytest.approx(0.1), pytest.approx(0.4)]


@pytest.mark.asyncio
async def test_dashscope_reranker_orders_chunks_by_score() -> None:
    client = DashScopeRerankClient(
        model="qwen3-rerank",
        api_key="test-key",
        max_concurrency=None,
    )
    mock_response = httpx.Response(
        200,
        json={
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.50},
                {"index": 1, "relevance_score": 0.10},
            ],
        },
        request=httpx.Request("POST", client.endpoint),
    )
    reranker = DashScopeReranker(client=client)
    chunks = [
        Chunk(content="first", metadata={"chunk_id": "0"}),
        Chunk(content="second", metadata={"chunk_id": "1"}),
        Chunk(content="third", metadata={"chunk_id": "2"}),
    ]

    with patch(
        "rag.reranker.dashscope_rerank_client.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        ranked = await reranker.arerank("query", chunks)

    assert [chunk.metadata["chunk_id"] for chunk in ranked] == ["2", "0", "1"]
    assert ranked[0].score == pytest.approx(0.95)


def test_make_reranker_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RERANK_MODEL_ID", raising=False)
    assert make_reranker(enabled=False) is None


def test_resolve_rerank_backend_prefers_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANK_BACKEND", "auto")
    monkeypatch.setenv("RERANK_MODEL_ID", "qwen3-rerank")
    assert resolve_rerank_backend() == "dashscope"


def test_make_reranker_uses_cross_encoder_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RERANK_MODEL_ID", raising=False)
    monkeypatch.setenv("RERANK_BACKEND", "cross_encoder")
    reranker = make_reranker(enabled=True)
    assert reranker is not None
    assert reranker.__class__.__name__ == "CrossEncoderReranker"
