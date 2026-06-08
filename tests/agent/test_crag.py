"""Tests for CRAG subgraph routing and metadata."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import AgentConfig, build_ReAct_agent, route_after_crag
from agent.metadata_schema import DEFAULT_RAG_TOOL_NAME
from agent.reflection.parsers import extract_rag_tool_results, split_rag_chunks
from agent.state import AgentState
from agent.subgraph.CRAG import (
    CragConfig,
    build_crag_subgraph,
    extract_rag_context,
    route_verdict,
)

pytestmark = pytest.mark.unit


def _rag_state(
    *,
    query: str = "What is RAG?",
    raw: str = "chunk one\n\n---\n\nchunk two",
    tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> AgentState:
    return {
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool_name, "args": {"query": query}, "id": "tc_rag"},
                ],
            ),
            ToolMessage(content=raw, tool_call_id="tc_rag"),
        ],
        "metadata": {},
    }


def test_split_rag_chunks():
    raw = "alpha\n\n---\n\nbeta"
    assert split_rag_chunks(raw) == ["alpha", "beta"]
    assert split_rag_chunks("") == []


def test_extract_rag_tool_results():
    hits = extract_rag_tool_results(_rag_state()["messages"])
    assert len(hits) == 1
    assert hits[0]["query"] == "What is RAG?"
    assert "chunk one" in hits[0]["raw"]


@pytest.mark.asyncio
async def test_extract_rag_context_skips_non_rag_tools():
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "math_tool", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="42", tool_call_id="tc1"),
        ],
        "metadata": {},
    }
    result = await extract_rag_context(state, rag_tool_name=DEFAULT_RAG_TOOL_NAME)
    assert result["metadata"]["crag_verdict"] == "skipped"
    assert result["metadata"]["crag_action"] == "use"


@pytest.mark.asyncio
async def test_extract_rag_context_records_attempt():
    result = await extract_rag_context(
        _rag_state(),
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
    )
    meta = result["metadata"]
    assert meta["rag_attempt"] == 1
    assert meta["rag_last_query"] == "What is RAG?"
    assert "chunk one" in meta["rag_last_raw"]


async def _score_all_correct(query: str, passages: list[str]) -> list[dict]:
    return [{"index": i, "label": "correct"} for i in range(len(passages))]


async def _score_first_incorrect(query: str, passages: list[str]) -> list[dict]:
    labels = [{"index": 0, "label": "incorrect"}]
    labels.extend({"index": i, "label": "ambiguous"} for i in range(1, len(passages)))
    return labels


@pytest.mark.asyncio
async def test_route_verdict_use_when_all_correct():
    state = _rag_state()
    state["metadata"] = {"rag_attempt": 1, "crag_labels": [{"index": 0, "label": "correct"}]}
    result = await route_verdict(state, max_rag_attempts=2)
    assert result["metadata"]["crag_verdict"] == "correct"
    assert result["metadata"]["crag_action"] == "use"


@pytest.mark.asyncio
async def test_route_verdict_requery_when_incorrect_and_under_limit():
    state = _rag_state()
    state["metadata"] = {
        "rag_attempt": 1,
        "rag_last_query": "What is RAG?",
        "crag_labels": [{"index": 0, "label": "incorrect"}],
    }
    result = await route_verdict(state, max_rag_attempts=2)
    assert result["metadata"]["crag_verdict"] == "incorrect"
    assert result["metadata"]["crag_action"] == "requery"
    assert "crag_requery_hint" in result["metadata"]


@pytest.mark.asyncio
async def test_route_verdict_degrade_at_attempt_limit():
    state = _rag_state()
    state["metadata"] = {
        "rag_attempt": 2,
        "crag_labels": [{"index": 0, "label": "incorrect"}],
    }
    result = await route_verdict(state, max_rag_attempts=2)
    assert result["metadata"]["crag_action"] == "degrade"


@pytest.mark.asyncio
async def test_crag_subgraph_skips_without_rag_tool():
    graph = build_crag_subgraph(CragConfig(score_fn=_score_all_correct))
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "math_tool", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="42", tool_call_id="tc1"),
        ],
        "metadata": {},
    }
    result = await graph.ainvoke(state)
    assert result["metadata"]["crag_verdict"] == "skipped"


async def _score_mixed_for_trim(query: str, passages: list[str]) -> list[dict]:
    labels = [{"index": 0, "label": "correct"}]
    labels.extend({"index": i, "label": "ambiguous"} for i in range(1, len(passages)))
    return labels


@pytest.mark.asyncio
async def test_crag_subgraph_trims_correct_passages():
    graph = build_crag_subgraph(CragConfig(score_fn=_score_mixed_for_trim))
    state = _rag_state(raw="keep me\n\n---\n\ndrop me")
    result = await graph.ainvoke(state)

    assert result["metadata"]["crag_action"] == "use"
    assert result["metadata"]["crag_verdict"] == "correct"
    tool_msg = result["messages"][-1]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.content == "keep me"


@pytest.mark.asyncio
async def test_crag_subgraph_requery_metadata():
    graph = build_crag_subgraph(CragConfig(score_fn=_score_first_incorrect))
    result = await graph.ainvoke(_rag_state())
    assert result["metadata"]["crag_action"] == "requery"
    assert result["metadata"]["crag_verdict"] == "incorrect"


def test_route_after_crag_always_returns_llm():
    assert route_after_crag({"messages": [], "metadata": {"crag_action": "requery"}}) == "llm"
    assert route_after_crag({"messages": [], "metadata": {"crag_verdict": "skipped"}}) == "llm"


def test_build_react_agent_with_crag_has_crag_node():
    class _FakeLLM:
        pass

    graph = build_ReAct_agent(
        AgentConfig(llm=_FakeLLM(), enable_crag=True),  # type: ignore[arg-type]
    )
    assert "crag_eval" in graph.get_graph().nodes


def test_build_agent_react_crag_pattern():
    class _FakeLLM:
        pass

    from agent.graph import build_agent

    graph = build_agent(
        AgentConfig(llm=_FakeLLM()),  # type: ignore[arg-type]
        pattern="react_crag",
    )
    assert "crag_eval" in graph.get_graph().nodes
