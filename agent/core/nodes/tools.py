"""Core ReAct nodes — tool execution."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from agent.core.edges.tool_calls import last_ai_message, tool_calls_from_ai
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
    """Execute all tool calls from the latest AIMessage.

    Uses :func:`last_ai_message` (not ``messages[-1]``) so a trailing
    HumanMessage/ToolMessage does not break execution. Returns an empty batch
    when there is no AIMessage or no pending tool calls.
    """
    ai_msg = last_ai_message(state["messages"])
    if ai_msg is None:
        return {"messages": []}

    tool_calls = tool_calls_from_ai(ai_msg)
    if not tool_calls:
        return {"messages": []}

    results: list[ToolMessage] = []
    for call in tool_calls:
        call_id = call.get("id")
        if not call_id:
            raise ValueError(
                f"Tool call {call.get('name')!r} is missing required 'id'"
            )
        result = await tool_box.ainvoke(
            name=str(call["name"]),
            args=dict(call.get("args") or {}),
        )
        results.append(
            ToolMessage(
                content=_tool_result_content(result),
                tool_call_id=call_id,
            )
        )
    return {"messages": results}
