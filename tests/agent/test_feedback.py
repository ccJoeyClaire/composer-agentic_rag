"""Tests for Feedback subgraph and graph wiring."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import AgentConfig, build_agent, build_ReAct_agent
from agent.state import AgentState
from agent.reflection.feedback import (
    FeedbackConfig,
    detect_feedback_node,
    heuristic_plan_feedback,
    plan_feedback_node,
    route_after_detect,
    rule_based_detect_feedback,
)

pytestmark = pytest.mark.unit


def test_rule_based_detect_correction():
    result = rule_based_detect_feedback("That's wrong, RAG is not that.")
    assert result["detected"] is True
    assert result["kind"] == "correction"


def test_rule_based_detect_clarify():
    result = rule_based_detect_feedback("What do you mean by embedding?")
    assert result["detected"] is True
    assert result["kind"] == "clarify"


def test_rule_based_detect_normal_question():
    result = rule_based_detect_feedback("What is vector search?")
    assert result["detected"] is False


def test_heuristic_plan_requery_after_retrieval():
    plan = heuristic_plan_feedback(
        "That's wrong, search again for hybrid retrieval.",
        {"rag_last_query": "RAG basics", "rag_last_raw": "old context"},
    )
    assert plan["action"] == "requery"
    assert plan["suggested_query"] is not None
    assert "feedback" not in (plan.get("hint") or "").lower() or plan.get("hint")


def test_heuristic_plan_clarify():
    plan = heuristic_plan_feedback("能解释一下吗？", {})
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
async def test_detect_feedback_marks_correction():
    state: AgentState = {
        "messages": [HumanMessage(content="不对，你答错了")],
        "metadata": {},
    }
    result = await detect_feedback_node(state, llm=None, detect_fn=None)
    assert result["metadata"]["feedback_detected"] is True
    assert result["metadata"]["feedback_kind"] == "correction"


@pytest.mark.asyncio
async def test_plan_feedback_sets_requery_metadata():
    state: AgentState = {
        "messages": [HumanMessage(content="That's wrong, try hybrid retrieval instead.")],
        "metadata": {
            "feedback_detected": True,
            "feedback_kind": "correction",
            "rag_last_query": "RAG basics",
            "rag_last_raw": "context",
        },
    }
    result = await plan_feedback_node(state, llm=None, plan_fn=None)
    meta = result["metadata"]
    assert meta["feedback_action"] == "requery"
    assert meta["feedback_suggested_query"]
    assert meta["feedback_hint"]


def test_route_after_detect():
    assert route_after_detect({"messages": [], "metadata": {"feedback_detected": True}}) == "plan_feedback"
    assert route_after_detect({"messages": [], "metadata": {"feedback_detected": False}}) == "continue"


@pytest.mark.asyncio
async def test_feedback_nodes_end_to_end():
    config = FeedbackConfig()
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
async def test_feedback_detect_clears_on_normal_question():
    config = FeedbackConfig()
    result = await detect_feedback_node(
        {"messages": [HumanMessage(content="What is chunking?")], "metadata": {}},
        llm=config.llm,
        detect_fn=config.detect_fn,
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
