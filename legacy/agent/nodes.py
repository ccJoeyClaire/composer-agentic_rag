from langchain_core.messages import ToolMessage

from agent.messages import messages_to_openai, openai_response_to_ai_message
from agent.state import AgentState
from llm.client import LLMClient
from tools.tool_box import ToolBox, ToolResult


def _tool_result_content(result: ToolResult) -> str:
    if result.error is not None:
        return result.error
    return str(result.output)


async def llm_node(
    state: AgentState,
    *,
    llm: LLMClient,
    tool_box: ToolBox,
    tool_calls: bool = True,
) -> dict:
    """Call LLM with current message history."""
    openai_messages = messages_to_openai(state["messages"])
    tools = tool_box.list_tools() if tool_calls else None
    response = await llm.arequest_llm(
        openai_messages,
        tool_calls=tool_calls,
        tools=tools,
    )
    return {"messages": [openai_response_to_ai_message(response)]}


async def tool_node(state: AgentState, *, tool_box: ToolBox) -> dict:
    """Execute all tool calls from the last LLM message."""
    last_msg = state["messages"][-1]
    results = []

    for call in last_msg.tool_calls:
        result = await tool_box.ainvoke(name=call["name"], args=call["args"])
        results.append(
            ToolMessage(
                content=_tool_result_content(result),
                tool_call_id=call["id"],
            )
        )
    return {"messages": results}
