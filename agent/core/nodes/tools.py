"""Core ReAct nodes — tool execution."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from agent.core.state import AgentState
from agent.core.tool_box import AgentToolBox
from tools.tool_box import ToolResult


def _tool_result_content(result: ToolResult) -> str:
    if result.error is not None:
        return result.error
    return str(result.output)


async def tool_node(
    state: AgentState,
    *,
    tool_box: AgentToolBox,
) -> dict[str, object]:
    """Execute all tool calls from the latest AIMessage."""
    last_msg = state["messages"][-1]
    results: list[ToolMessage] = []

    for call in last_msg.tool_calls:
        result = await tool_box.ainvoke(name=call["name"], args=call["args"])
        results.append(
            ToolMessage(
                content=_tool_result_content(result),
                tool_call_id=call["id"],
            )
        )
    return {"messages": results}
