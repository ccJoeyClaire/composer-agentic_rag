"""Tests for agent graph builder and routing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

from agent.builder import build_agent
from agent.capabilities.human_feedback.tool import CLARIFICATION_TOOL_NAME
from agent.config import AgentConfig
from agent.core.edges.after_llm import route_after_llm
from agent.core.edges.after_tools import route_after_tools
from agent.core.edges.names import NodeName
from agent.core.constants import DEFAULT_RAG_TOOL_NAME
from agent.core.state import AgentState

pytestmark = pytest.mark.unit


@pytest.fixture
def llm_stub() -> MagicMock:
    return MagicMock()


def _config(llm_stub: MagicMock, **kwargs: object) -> AgentConfig:
    return AgentConfig(llm=llm_stub, **kwargs)  # type: ignore[arg-type]


class TestRouteAfterLlm:
    def test_no_tool_calls_goes_to_end(self, llm_stub: MagicMock) -> None:
        state: AgentState = {
            "messages": [AIMessage(content="Final answer.", tool_calls=[])]
        }
        assert route_after_llm(state, agent_config=_config(llm_stub)) == END

    def test_rag_tool_calls_with_router_enabled(self, llm_stub: MagicMock) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": DEFAULT_RAG_TOOL_NAME,
                            "args": {"query": "test"},
                            "id": "c1",
                        }
                    ],
                )
            ]
        }
        config = _config(llm_stub, enable_rag_profile_router=True)
        assert route_after_llm(state, agent_config=config) == NodeName.RAG_PROFILE_ROUTER

    def test_rag_tool_calls_router_disabled_goes_to_tools(
        self,
        llm_stub: MagicMock,
    ) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": DEFAULT_RAG_TOOL_NAME,
                            "args": {"query": "test"},
                            "id": "c1",
                        }
                    ],
                )
            ]
        }
        assert route_after_llm(state, agent_config=_config(llm_stub)) == NodeName.TOOLS

    def test_web_tool_calls_goes_to_tools(self, llm_stub: MagicMock) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "tavily_search", "args": {"query": "x"}, "id": "c1"},
                    ],
                )
            ]
        }
        assert route_after_llm(state, agent_config=_config(llm_stub)) == NodeName.TOOLS


class TestRouteAfterTools:
    def test_rag_batch_with_gate_enabled(self, llm_stub: MagicMock) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": DEFAULT_RAG_TOOL_NAME,
                            "args": {"query": "q"},
                            "id": "c1",
                        }
                    ],
                ),
                ToolMessage(content="passages", tool_call_id="c1"),
            ]
        }
        config = _config(llm_stub, enable_retrieval_gate=True)
        assert route_after_tools(state, agent_config=config) == NodeName.RETRIEVAL_GATE

    def test_web_batch_goes_to_llm(self, llm_stub: MagicMock) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "tavily_search", "args": {"query": "q"}, "id": "c1"},
                    ],
                ),
                ToolMessage(content="results", tool_call_id="c1"),
            ]
        }
        assert route_after_tools(state, agent_config=_config(llm_stub)) == NodeName.LLM

    def test_clarification_batch_goes_to_llm(self, llm_stub: MagicMock) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": CLARIFICATION_TOOL_NAME,
                            "args": {"question": "Which dataset?"},
                            "id": "c1",
                        }
                    ],
                ),
                ToolMessage(content="user reply", tool_call_id="c1"),
            ]
        }
        assert route_after_tools(state, agent_config=_config(llm_stub)) == NodeName.LLM


class TestBuildAgent:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"enable_rag_profile_router": True},
            {"enable_retrieval_gate": True},
            {"enable_human_feedback": True},
            {
                "enable_rag_profile_router": True,
                "enable_retrieval_gate": True,
                "enable_human_feedback": True,
            },
        ],
    )
    def test_graph_compiles(self, llm_stub: MagicMock, kwargs: dict) -> None:
        graph = build_agent(_config(llm_stub, **kwargs))
        assert graph is not None

    def test_base_graph_has_llm_entry(self, llm_stub: MagicMock) -> None:
        graph = build_agent(_config(llm_stub))
        node_names = set(graph.get_graph().nodes)
        assert NodeName.LLM in node_names
        assert "__start__" in node_names
