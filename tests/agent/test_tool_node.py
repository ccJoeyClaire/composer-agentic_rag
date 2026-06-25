"""Tests for agent.core.nodes.tools."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.core.nodes.tools import tool_node
from agent.core.state import AgentState
from agent.core.tool_box import AgentToolBox
from tools.tool_box import ToolBox, ToolResult

pytestmark = pytest.mark.unit


class _EchoToolBox(ToolBox):
    async def ainvoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(name=name, args=args, output=f"ok:{args.get('query', '')}")


@pytest.fixture
def tool_box() -> AgentToolBox:
    return AgentToolBox(inner=_EchoToolBox())


async def test_tool_node_returns_tool_messages_with_ids(
    tool_box: AgentToolBox,
) -> None:
    state: AgentState = {
        "messages": [
            AIMessage(
                content="Searching.",
                tool_calls=[
                    {
                        "name": "RAG_search_tool",
                        "args": {"query": "test"},
                        "id": "tc1",
                    }
                ],
            )
        ]
    }
    result = await tool_node(state, tool_box=tool_box)
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert msgs[0].tool_call_id == "tc1"
    assert msgs[0].content == "ok:test"


async def test_tool_node_uses_last_ai_message_not_trailing_human(
    tool_box: AgentToolBox,
) -> None:
    """Trailing HumanMessage must not break tool execution."""
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "RAG_search_tool", "args": {"query": "q"}, "id": "tc1"}
                ],
            ),
            HumanMessage(content="ignored follow-up"),
        ]
    }
    result = await tool_node(state, tool_box=tool_box)
    assert len(result["messages"]) == 1
    assert result["messages"][0].tool_call_id == "tc1"


async def test_tool_node_returns_empty_when_last_is_tool_message(
    tool_box: AgentToolBox,
) -> None:
    state: AgentState = {
        "messages": [
            AIMessage(content="", tool_calls=[]),
            ToolMessage(content="done", tool_call_id="tc1"),
        ]
    }
    result = await tool_node(state, tool_box=tool_box)
    assert result["messages"] == []


async def test_tool_node_raises_when_call_id_missing(tool_box: AgentToolBox) -> None:
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "bad", "args": {}}],
            )
        ]
    }
    with pytest.raises(ValueError, match="missing required 'id'"):
        await tool_node(state, tool_box=tool_box)


async def test_tool_node_surfaces_errors() -> None:
    class _ErrorToolBox(ToolBox):
        async def ainvoke(self, name: str, args: dict[str, Any]) -> ToolResult:
            return ToolResult(name=name, args=args, error="boom")

    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "bad", "args": {}, "id": "e1"}],
            )
        ]
    }
    result = await tool_node(
        state,
        tool_box=AgentToolBox(inner=_ErrorToolBox()),
    )
    assert result["messages"][0].content == "boom"
