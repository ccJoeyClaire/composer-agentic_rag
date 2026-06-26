"""Tests for Feedback subgraph and graph wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from legacy.agent.graph import AgentConfig, build_agent, build_ReAct_agent
from legacy.agent.state import AgentState
from legacy.agent.reflection.feedback import (
    FeedbackConfig,
    default_detect_feedback,
    default_plan_feedback,
    detect_feedback_node,
    plan_feedback_node,
    route_after_detect,
)

pytestmark = pytest.mark.unit


def _mock_llm_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps(payload)
    return response


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.arequest_llm = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_default_detect_feedback_correction(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {"detected": True, "kind": "correction"}
    )
    result = await default_detect_feedback(mock_llm, "That's wrong, RAG is not that.")
    assert result["detected"] is True
    assert result["kind"] == "correction"
    mock_llm.arequest_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_detect_feedback_clarify(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {"detected": True, "kind": "clarify"}
    )
    result = await default_detect_feedback(mock_llm, "What do you mean by embedding?")
    assert result["detected"] is True
    assert result["kind"] == "clarify"


@pytest.mark.asyncio
async def test_default_detect_feedback_normal_question(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {"detected": False, "kind": None}
    )
    result = await default_detect_feedback(mock_llm, "What is vector search?")
    assert result["detected"] is False


@pytest.mark.asyncio
async def test_default_plan_requery_after_retrieval(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {
            "action": "requery",
            "kind": "correction",
            "suggested_query": "hybrid retrieval",
            "hint": "Search again with a refined query.",
        }
    )
    plan = await default_plan_feedback(
        mock_llm,
        "That's wrong, search again for hybrid retrieval.",
        {"rag_last_query": "RAG basics", "rag_last_raw": "old context"},
    )
    assert plan["action"] == "requery"
    assert plan["suggested_query"] == "hybrid retrieval"


@pytest.mark.asyncio
async def test_default_plan_clarify(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {
            "action": "clarify",
            "kind": "clarify",
            "suggested_query": None,
            "hint": "Ask what was unclear.",
        }
    )
    plan = await default_plan_feedback(mock_llm, "能解释一下吗？", {})
    assert plan["action"] == "clarify"


@pytest.mark.asyncio
async def test_detect_feedback_clears_when_last_message_not_human():
    state: AgentState = {
        "messages": [AIMessage(content="previous answer")],
        "metadata": {"feedback_detected": True},
    }
    result = await detect_feedback_node(state, llm=None, detect_fn=None)
    assert result["metadata"]["feedback_detected"] is False


@pytest.mark.asyncio
async def test_detect_feedback_marks_correction(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {"detected": True, "kind": "correction"}
    )
    state: AgentState = {
        "messages": [HumanMessage(content="不对，你答错了")],
        "metadata": {},
    }
    result = await detect_feedback_node(state, llm=mock_llm, detect_fn=None)
    assert result["metadata"]["feedback_detected"] is True
    assert result["metadata"]["feedback_kind"] == "correction"


@pytest.mark.asyncio
async def test_detect_feedback_skips_without_llm():
    state: AgentState = {
        "messages": [HumanMessage(content="不对，你答错了")],
        "metadata": {},
    }
    result = await detect_feedback_node(state, llm=None, detect_fn=None)
    assert result["metadata"]["feedback_detected"] is False


@pytest.mark.asyncio
async def test_plan_feedback_sets_requery_metadata(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {
            "action": "requery",
            "kind": "correction",
            "suggested_query": "hybrid retrieval",
            "hint": "Refine the search query.",
        }
    )
    state: AgentState = {
        "messages": [HumanMessage(content="That's wrong, try hybrid retrieval instead.")],
        "metadata": {
            "feedback_detected": True,
            "feedback_kind": "correction",
            "rag_last_query": "RAG basics",
            "rag_last_raw": "context",
        },
    }
    result = await plan_feedback_node(state, llm=mock_llm, plan_fn=None)
    meta = result["metadata"]
    assert meta["feedback_action"] == "requery"
    assert meta["feedback_suggested_query"]
    assert meta["feedback_hint"]


def test_route_after_detect():
    assert route_after_detect({"messages": [], "metadata": {"feedback_detected": True}}) == "plan_feedback"
    assert route_after_detect({"messages": [], "metadata": {"feedback_detected": False}}) == "continue"


@pytest.mark.asyncio
async def test_feedback_nodes_end_to_end(mock_llm: MagicMock):
    mock_llm.arequest_llm = AsyncMock(
        side_effect=[
            _mock_llm_response({"detected": True, "kind": "correction"}),
            _mock_llm_response(
                {
                    "action": "requery",
                    "kind": "correction",
                    "suggested_query": "向量数据库",
                    "hint": "Re-search the vector database.",
                }
            ),
        ]
    )
    config = FeedbackConfig(llm=mock_llm)
    state: AgentState = {
        "messages": [HumanMessage(content="不对，请重新检索向量数据库")],
        "metadata": {
            "rag_last_query": "database",
            "rag_last_raw": "old answer context",
        },
    }
    detected = await detect_feedback_node(state, llm=config.llm, detect_fn=config.detect_fn)
    assert detected["metadata"]["feedback_detected"] is True
    assert route_after_detect(detected) == "plan_feedback"

    state["metadata"] = detected["metadata"]
    planned = await plan_feedback_node(state, llm=config.llm, plan_fn=config.plan_fn)
    meta = planned["metadata"]
    assert meta["feedback_action"] == "requery"
    assert meta["feedback_suggested_query"]


@pytest.mark.asyncio
async def test_feedback_detect_clears_on_normal_question(mock_llm: MagicMock):
    mock_llm.arequest_llm.return_value = _mock_llm_response(
        {"detected": False, "kind": None}
    )
    result = await detect_feedback_node(
        {"messages": [HumanMessage(content="What is chunking?")], "metadata": {}},
        llm=mock_llm,
        detect_fn=None,
    )
    assert result["metadata"]["feedback_detected"] is False
    assert route_after_detect(result) == "continue"


def test_build_agent_feedback_pattern():
    class _FakeLLM:
        pass

    graph = build_agent(
        AgentConfig(llm=_FakeLLM()),  # type: ignore[arg-type]
        pattern="react_feedback",
    )
    nodes = graph.get_graph().nodes
    assert "detect_feedback" in nodes
    assert "plan_feedback" in nodes


def test_build_agent_all_pattern():
    class _FakeLLM:
        pass

    graph = build_agent(
        AgentConfig(llm=_FakeLLM()),  # type: ignore[arg-type]
        pattern="react_all",
    )
    nodes = graph.get_graph().nodes
    assert "detect_feedback" in nodes
    assert "plan_feedback" in nodes
    assert "self_rag_pre" in nodes
    assert "self_rag_post" in nodes
    assert "crag_eval" in nodes


def test_build_react_agent_feedback_entry_before_self_rag():
    class _FakeLLM:
        pass

    graph = build_ReAct_agent(
        AgentConfig(llm=_FakeLLM(), enable_feedback=True, enable_self_rag=True),  # type: ignore[arg-type]
    )
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("__start__", "detect_feedback") in edges
    assert ("plan_feedback", "self_rag_pre") in edges


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/agent/test_feedback.py -v
# ================================================================================================================
