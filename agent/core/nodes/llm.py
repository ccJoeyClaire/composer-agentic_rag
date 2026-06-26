"""Core ReAct nodes — LLM invocation."""

from __future__ import annotations

from agent.capabilities.retrieval_gate.gate_context import (
    build_gate_context_message,
    prepare_messages_for_llm,
)
from agent.core.constants import DEFAULT_RAG_TOOL_NAME
from agent.core.edges.tool_calls import last_batch_included_rag
from agent.core.state import AgentState, get_metadata
from agent.core.tool_box import AgentToolBox
from agent.messages import (
    messages_to_openai,
    openai_response_to_ai_message,
    prepare_rag_context_for_llm,
    resolve_rag_context_max_chunks,
)
from llm.client import LLMClient


async def llm_node(
    state: AgentState,
    *,
    llm: LLMClient,
    tool_box: AgentToolBox,
    tool_calls: bool = True,
    enable_retrieval_gate: bool = False,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
    rag_context_max_chunks: int | None = None,
) -> dict[str, object]:
    """Call the LLM with the current message history and optional tools."""
    view = list(state["messages"])
    if enable_retrieval_gate and last_batch_included_rag(
        state,
        rag_tool_name=rag_tool_name,
    ):
        gate_message = build_gate_context_message(get_metadata(state))
        view = prepare_messages_for_llm(view, gate_message=gate_message)

    max_chunks = resolve_rag_context_max_chunks(rag_context_max_chunks)
    view = prepare_rag_context_for_llm(
        view,
        rag_tool_name=rag_tool_name,
        max_chunks=max_chunks,
    )

    openai_messages = messages_to_openai(view)
    tools = tool_box.list_tools() if tool_calls else None
    response = await llm.arequest_llm(
        openai_messages,
        tool_calls=tool_calls,
        tools=tools,
    )
    return {"messages": [openai_response_to_ai_message(response)]}
