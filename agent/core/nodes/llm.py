"""Core ReAct nodes — LLM invocation."""

from __future__ import annotations

from agent.core.state import AgentState
from agent.core.tool_box import AgentToolBox
from agent.messages import messages_to_openai, openai_response_to_ai_message
from llm.client import LLMClient


async def llm_node(
    state: AgentState,
    *,
    llm: LLMClient,
    tool_box: AgentToolBox,
    tool_calls: bool = True,
) -> dict[str, object]:
    """Call the LLM with the current message history and optional tools."""
    openai_messages = messages_to_openai(state["messages"])
    tools = tool_box.list_tools() if tool_calls else None
    response = await llm.arequest_llm(
        openai_messages,
        tool_calls=tool_calls,
        tools=tools,
    )
    return {"messages": [openai_response_to_ai_message(response)]}
