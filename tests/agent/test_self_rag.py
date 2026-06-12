"""Tests for Self-RAG pre/post subgraphs and graph wiring."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent.graph import (
    AgentConfig,
    build_ReAct_agent,
    build_agent,
    if_after_llm,
    route_after_self_rag_post,
)
from agent.metadata_schema import DEFAULT_RAG_TOOL_NAME
from agent.state import AgentState
from agent.reflection.self_rag import (
    SelfRagConfig,
    rule_based_need_retrieve,
    self_rag_post_node,
    self_rag_pre_node,
)

pytestmark = pytest.mark.unit


def test_rule_based_need_retrieve():
    assert rule_based_need_retrieve("What is retrieval augmented generation?") is True
    assert rule_based_need_retrieve("你好") is False
    assert rule_based_need_retrieve("hi") is False


@pytest.mark.asyncio
async def test_self_rag_pre_sets_need_retrieve():
    state: AgentState = {
        "messages": [HumanMessage(content="What is vector search?")],
        "metadata": {},
    }
    result = await self_rag_pre_node(
        state,
        llm=None,
        classify_fn=None,
        max_rag_attempts=2,
    )
    assert result["metadata"]["self_rag_need_retrieve"] is True
    assert result["metadata"]["self_rag_retry_allowed"] is True


@pytest.mark.asyncio
async def test_self_rag_pre_skips_without_human_message():
    state: AgentState = {"messages": [AIMessage(content="hello")], "metadata": {}}
    result = await self_rag_pre_node(
        state,
        llm=None,
        classify_fn=None,
        max_rag_attempts=2,
    )
    assert result["metadata"]["self_rag_need_retrieve"] is None


async def _always_grounded(query: str, context: str, answer: str) -> bool:
    return True


async def _never_grounded(query: str, context: str, answer: str) -> bool:
    return False


@pytest.mark.asyncio
async def test_self_rag_post_skips_without_context():
    state: AgentState = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="answer without retrieval"),
        ],
        "metadata": {},
    }
    result = await self_rag_post_node(
        state,
        llm=None,
        grounded_fn=_always_grounded,
        max_rag_attempts=2,
    )
    assert result["metadata"]["self_rag_grounded"] is None


@pytest.mark.asyncio
async def test_self_rag_post_marks_grounded():
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is RAG?"),
            AIMessage(content="RAG combines retrieval with generation."),
        ],
        "metadata": {
            "rag_last_query": "What is RAG?",
            "rag_last_raw": "RAG combines retrieval with generation.",
        },
    }
    result = await self_rag_post_node(
        state,
        llm=None,
        grounded_fn=_always_grounded,
        max_rag_attempts=2,
    )
    assert result["metadata"]["self_rag_grounded"] is True


@pytest.mark.asyncio
async def test_self_rag_post_adds_retry_hint_when_ungrounded():
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is RAG?"),
            AIMessage(content="It is a cooking technique."),
        ],
        "metadata": {
            "rag_last_query": "What is RAG?",
            "rag_last_raw": "Retrieval augmented generation for LLMs.",
            "rag_attempt": 1,
            "self_rag_retry_allowed": True,
        },
    }
    result = await self_rag_post_node(
        state,
        llm=None,
        grounded_fn=_never_grounded,
        max_rag_attempts=2,
    )
    assert result["metadata"]["self_rag_grounded"] is False
    assert "self_rag_retry_hint" in result["metadata"]


@pytest.mark.asyncio
async def test_self_rag_pre_node_runs_with_config():
    config = SelfRagConfig()
    result = await self_rag_pre_node(
        {"messages": [HumanMessage(content="Explain embeddings.")], "metadata": {}},
        llm=config.llm,
        classify_fn=config.classify_fn,
        max_rag_attempts=config.max_rag_attempts,
    )
    assert result["metadata"]["self_rag_need_retrieve"] is True


@pytest.mark.asyncio
async def test_self_rag_post_node_runs_with_config():
    config = SelfRagConfig(grounded_fn=_always_grounded)
    result = await self_rag_post_node(
        {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="supported answer"),
            ],
            "metadata": {
                "rag_last_query": "q",
                "rag_last_raw": "supported answer",
            },
        },
        llm=config.llm,
        grounded_fn=config.grounded_fn,
        max_rag_attempts=config.max_rag_attempts,
    )
    assert result["metadata"]["self_rag_grounded"] is True


def test_if_after_llm_routes_to_self_rag_post():
    state: AgentState = {"messages": [AIMessage(content="final answer", tool_calls=[])]}
    assert if_after_llm(state, use_self_rag=True) == "self_rag_post"
    assert if_after_llm(state, use_self_rag=False) == END


def test_if_after_llm_routes_tool_calls_to_tools():
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": DEFAULT_RAG_TOOL_NAME, "args": {}, "id": "1"}],
            )
        ]
    }
    assert if_after_llm(state, use_self_rag=True) == "tools"


def test_route_after_self_rag_post_retries_when_ungrounded():
    state: AgentState = {
        "messages": [],
        "metadata": {
            "self_rag_grounded": False,
            "self_rag_retry_allowed": True,
            "rag_attempt": 1,
            "max_rag_attempts": 2,
        },
    }
    assert route_after_self_rag_post(state) == "llm"


def test_route_after_self_rag_post_ends_when_grounded():
    state: AgentState = {
        "messages": [],
        "metadata": {"self_rag_grounded": True},
    }
    assert route_after_self_rag_post(state) == END


def test_build_agent_self_rag_pattern():
    class _FakeLLM:
        pass

    graph = build_agent(
        AgentConfig(llm=_FakeLLM()),  # type: ignore[arg-type]
        pattern="react_self_rag",
    )
    nodes = graph.get_graph().nodes
    assert "self_rag_pre" in nodes
    assert "self_rag_post" in nodes


def test_build_agent_full_pattern_has_crag_and_self_rag():
    class _FakeLLM:
        pass

    graph = build_agent(
        AgentConfig(llm=_FakeLLM()),  # type: ignore[arg-type]
        pattern="react_full",
    )
    nodes = graph.get_graph().nodes
    assert "self_rag_pre" in nodes
    assert "self_rag_post" in nodes
    assert "crag_eval" in nodes


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/agent/test_self_rag.py -v
# ================================================================================================================
