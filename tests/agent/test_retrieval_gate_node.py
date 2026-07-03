"""Tests for retrieval_gate scoring and node behavior."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.capabilities.retrieval_gate.config import (
    DEFAULT_PASS_THRESHOLD,
    RetrievalGateConfig,
)
from agent.capabilities.retrieval_gate.metadata import (
    GATE_EVIDENCE_SOURCES_KEY,
    GATE_ISSUES_KEY,
    GATE_PASSAGES_SUMMARY_KEY,
    GATE_VERDICT_KEY,
)
from agent.capabilities.retrieval_gate.node import retrieval_gate_node
from agent.capabilities.retrieval_gate.rag_context import (
    RAG_PASSAGE_SEPARATOR,
    split_rag_passages,
)
from agent.core.tool_box import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME
from agent.core.state import AgentState

pytestmark = pytest.mark.unit


def test_split_rag_passages() -> None:
    raw = f"first{RAG_PASSAGE_SEPARATOR}second"
    assert split_rag_passages(raw) == ["first", "second"]


def test_split_rag_passages_parses_json_tool_output() -> None:
    raw = json.dumps(
        [
            {"content": "first", "score": 0.5},
            {"content": "second", "score": 0.3},
        ]
    )
    assert split_rag_passages(raw) == ["first", "second"]


@pytest.mark.asyncio
async def test_retrieval_gate_node_uses_user_query_for_scoring() -> None:
    passage_a = "RAG combines retrieval with LLMs."
    passage_b = "Paris weather is sunny."
    raw = f"{passage_a}{RAG_PASSAGE_SEPARATOR}{passage_b}"
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is RAG?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DEFAULT_RAG_TOOL_NAME,
                        "args": {"query": "rewritten query"},
                        "id": "tc1",
                    }
                ],
            ),
            ToolMessage(content=raw, tool_call_id="tc1"),
        ],
        "metadata": {},
    }

    async def fake_score(
        _state: AgentState,
        query: str,
        passages: list[str],
    ) -> list[float]:
        assert query == "What is RAG?"
        assert passages == [passage_a, passage_b]
        return [0.91, 0.12]

    config = RetrievalGateConfig(
        score_fn=fake_score,
        pass_threshold=DEFAULT_PASS_THRESHOLD,
    )
    patch = await retrieval_gate_node(state, capability_config=config)
    metadata = patch["metadata"]
    assert metadata[GATE_VERDICT_KEY] == "pass"
    assert metadata[GATE_EVIDENCE_SOURCES_KEY] == ["rag"]


@pytest.mark.asyncio
async def test_retrieval_gate_node_scores_web_batch() -> None:
    web_raw = json.dumps(
        {
            "results": [
                {"title": "Titanic", "content": "James Cameron directed Titanic."},
            ]
        }
    )
    state: AgentState = {
        "messages": [
            HumanMessage(content="Who directed Titanic?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DEFAULT_WEB_TOOL_NAME,
                        "args": {"query": "Titanic director"},
                        "id": "tc_web",
                    }
                ],
            ),
            ToolMessage(content=web_raw, tool_call_id="tc_web"),
        ],
        "metadata": {GATE_VERDICT_KEY: "low_quality"},
    }

    async def fake_score(
        _state: AgentState,
        query: str,
        passages: list[str],
    ) -> list[float]:
        assert query == "Who directed Titanic?"
        assert len(passages) == 1
        return [0.88]

    patch = await retrieval_gate_node(
        state,
        capability_config=RetrievalGateConfig(score_fn=fake_score),
    )
    metadata = patch["metadata"]
    assert metadata[GATE_VERDICT_KEY] == "pass"
    assert metadata[GATE_EVIDENCE_SOURCES_KEY] == ["web"]


def test_compute_gate_verdict_pass_at_default_threshold() -> None:
    from agent.capabilities.retrieval_gate.verdict import compute_gate_verdict

    verdict, issues = compute_gate_verdict(
        ["a", "b"],
        [0.2, 0.55],
        pass_threshold=DEFAULT_PASS_THRESHOLD,
    )
    assert verdict == "pass"
    assert issues == []


def test_compute_gate_verdict_low_quality_below_threshold() -> None:
    from agent.capabilities.retrieval_gate.verdict import compute_gate_verdict

    verdict, issues = compute_gate_verdict(
        ["a", "b"],
        [0.2, 0.34],
        pass_threshold=DEFAULT_PASS_THRESHOLD,
    )
    assert verdict == "low_quality"
    assert issues
    assert "0.34" in issues[0]


def test_compute_gate_verdict_error_on_score_count_mismatch() -> None:
    from agent.capabilities.retrieval_gate.verdict import compute_gate_verdict

    verdict, issues = compute_gate_verdict(
        ["a", "b"],
        [0.9],
        pass_threshold=DEFAULT_PASS_THRESHOLD,
    )
    assert verdict == "error"
    assert issues
    assert "scoring failed" in issues[0]


@pytest.mark.asyncio
async def test_retrieval_gate_node_retries_scoring_on_error() -> None:
    passage_a = "RAG combines retrieval with LLMs."
    passage_b = "Paris weather is sunny."
    raw = f"{passage_a}{RAG_PASSAGE_SEPARATOR}{passage_b}"
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is RAG?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DEFAULT_RAG_TOOL_NAME,
                        "args": {"query": "What is RAG?"},
                        "id": "tc1",
                    }
                ],
            ),
            ToolMessage(content=raw, tool_call_id="tc1"),
        ],
        "metadata": {},
    }
    calls = 0

    async def flaky_score(
        _state: AgentState,
        query: str,
        passages: list[str],
    ) -> list[float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [0.91]
        return [0.91, 0.12]

    config = RetrievalGateConfig(
        score_fn=flaky_score,
        pass_threshold=DEFAULT_PASS_THRESHOLD,
        max_scoring_retries=2,
    )
    patch = await retrieval_gate_node(state, capability_config=config)
    metadata = patch["metadata"]
    assert calls == 2
    assert metadata[GATE_VERDICT_KEY] == "pass"
    assert "error" not in patch


@pytest.mark.asyncio
async def test_retrieval_gate_node_empty_when_no_scorable_batch() -> None:
    state: AgentState = {
        "messages": [AIMessage(content="hello")],
        "metadata": {},
    }
    patch = await retrieval_gate_node(state, capability_config=RetrievalGateConfig())
    metadata = patch["metadata"]
    assert metadata[GATE_VERDICT_KEY] == "empty"
