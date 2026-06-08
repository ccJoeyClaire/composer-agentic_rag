"""Tests for agent graph routing and nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.nodes import llm_node, tool_node
from agent.state import AgentState
from tools.tool_box import ToolResult

try:
    from agent.graph import if_tool_calls

    _GRAPH_IMPORT_ERROR = None
except TypeError as exc:
    if_tool_calls = None  # type: ignore[assignment,misc]
    _GRAPH_IMPORT_ERROR = exc

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _skip_graph_if_broken(request):
    if _GRAPH_IMPORT_ERROR is None:
        return
    if request.node.get_closest_marker("needs_graph_import"):
        pytest.skip(f"agent.graph import blocked: {_GRAPH_IMPORT_ERROR}")


needs_graph = pytest.mark.needs_graph_import


@needs_graph
def test_if_tool_calls_routes_to_tools():
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "rag_search", "args": {"query": "test"}, "id": "call_1"},
                ],
            )
        ]
    }
    assert if_tool_calls(state) == "tools"


@needs_graph
def test_if_tool_calls_routes_to_end_when_no_tools():
    state: AgentState = {
        "messages": [AIMessage(content="Done.", tool_calls=[])]
    }
    assert if_tool_calls(state) != "tools"


@needs_graph
def test_if_tool_calls_routes_to_end_for_human_message():
    state: AgentState = {"messages": [HumanMessage(content="Hi")]}
    assert if_tool_calls(state) != "tools"


class _FakeToolBox:
    def __init__(self, output="ok"):
        self.output = output
        self.last_name = None
        self.last_args = None

    def list_tools(self):
        return []

    async def ainvoke(self, name: str, args: dict) -> ToolResult:
        self.last_name = name
        self.last_args = args
        return ToolResult(name=name, args=args, output=self.output)


@pytest.mark.asyncio
async def test_tool_node_returns_tool_messages():
    tool_box = _FakeToolBox(output="chunk text")
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "rag_search", "args": {"query": "RAG"}, "id": "tc1"},
                ],
            )
        ]
    }
    result = await tool_node(state, tool_box=tool_box)

    assert tool_box.last_name == "rag_search"
    assert tool_box.last_args == {"query": "RAG"}
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert msgs[0].content == "chunk text"
    assert msgs[0].tool_call_id == "tc1"


@pytest.mark.asyncio
async def test_tool_node_surfaces_errors():
    class _ErrorToolBox:
        def list_tools(self):
            return []

        async def ainvoke(self, name: str, args: dict) -> ToolResult:
            return ToolResult(name=name, args=args, error="tool failed")

    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "bad", "args": {}, "id": "e1"}],
            )
        ]
    }
    result = await tool_node(state, tool_box=_ErrorToolBox())
    assert result["messages"][0].content == "tool failed"


@pytest.mark.asyncio
async def test_llm_node_passes_tools_to_client():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "answer"
    mock_response.tool_calls = None
    mock_llm.arequest_llm = AsyncMock(return_value=mock_response)

    tool_box = _FakeToolBox()
    state: AgentState = {"messages": [HumanMessage(content="question")]}

    result = await llm_node(
        state,
        llm=mock_llm,
        tool_box=tool_box,
        tool_calls=True,
    )

    mock_llm.arequest_llm.assert_awaited_once()
    kwargs = mock_llm.arequest_llm.await_args.kwargs
    assert kwargs["tool_calls"] is True
    assert kwargs["tools"] == []
    assert "messages" in result
