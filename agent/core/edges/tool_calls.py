"""Helpers for inspecting tool calls on the message history."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent.core.constants import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME
from agent.core.state import AgentState


def last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Return the most recent AIMessage, or None."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def tool_calls_from_ai(ai_msg: AIMessage) -> list[dict[str, Any]]:
    """Normalize ``tool_calls`` on an AIMessage to a list (empty when absent)."""
    return list(getattr(ai_msg, "tool_calls", None) or [])


def preceding_ai_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """Tool calls from the AIMessage that triggered the latest tool batch."""
    messages = state["messages"]
    if not messages:
        return []

    if isinstance(messages[-1], ToolMessage):
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                return tool_calls_from_ai(msg)
        return []

    last = messages[-1]
    if isinstance(last, AIMessage):
        return tool_calls_from_ai(last)
    return []


def has_tool_named(tool_calls: list[dict[str, Any]], name: str) -> bool:
    """True when any tool call targets *name*."""
    return any(call.get("name") == name for call in tool_calls)


def has_rag_tool_call(
    tool_calls: list[dict[str, Any]],
    *,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> bool:
    """True when the batch includes a RAG search tool call."""
    return has_tool_named(tool_calls, rag_tool_name)


def last_batch_included_rag(
    state: AgentState,
    *,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> bool:
    """True when the tool batch just executed included RAG."""
    return has_rag_tool_call(
        preceding_ai_tool_calls(state),
        rag_tool_name=rag_tool_name,
    )



# =======================example usage=======================

if __name__ == "__main__":
    from agent.capabilities.human_feedback.tool import CLARIFICATION_TOOL_NAME

    def section(title: str) -> None:
        print(f"\n=== {title} ===")

    rag_call = {
        "name": DEFAULT_RAG_TOOL_NAME,
        "args": {"query": "capital of France"},
        "id": "call_rag",
    }
    clarify_call = {
        "name": CLARIFICATION_TOOL_NAME,
        "args": {"question": "Which France?", "timing": "pre_retrieval"},
        "id": "call_clarify",
    }
    web_call = {
        "name": DEFAULT_WEB_TOOL_NAME,
        "args": {"query": "France capital"},
        "id": "call_web",
    }

    ai_rag = AIMessage(
        content="I'll search the knowledge base.",
        tool_calls=[rag_call],
    )
    ai_clarify = AIMessage(
        content="I need more detail.",
        tool_calls=[clarify_call],
    )
    ai_web = AIMessage(
        content="I'll search the web.",
        tool_calls=[web_call],
    )
    ai_done = AIMessage(content="The capital of France is Paris.", tool_calls=[])
    tool_rag = ToolMessage(content="Paris is the capital.", tool_call_id="call_rag")
    tool_clarify = ToolMessage(
        content="clarification_requested:pre_retrieval:Which France?",
        tool_call_id="call_clarify",
    )

    section("after_llm — AI just requested RAG")
    state_after_llm_rag: AgentState = {"messages": [ai_rag]}
    ai_msg = last_ai_message(state_after_llm_rag["messages"])
    calls = tool_calls_from_ai(ai_msg) if ai_msg is not None else []
    print("ai_msg:", ai_msg)
    print("tool_calls:", calls)
    print("has_rag_tool_call:", has_rag_tool_call(calls))

    section("after_llm — AI finished without tools")
    state_after_llm_done: AgentState = {"messages": [ai_done]}
    ai_msg = last_ai_message(state_after_llm_done["messages"])
    calls = tool_calls_from_ai(ai_msg) if ai_msg is not None else []
    print("ai_msg:", ai_msg)
    print("tool_calls:", calls)
    print("route would be END (no tool calls)")

    section("after_tools — RAG tool just returned")
    state_after_tools_rag: AgentState = {"messages": [ai_rag, tool_rag]}
    batch = preceding_ai_tool_calls(state_after_tools_rag)
    print("preceding_ai_tool_calls:", batch)
    print("last_batch_included_rag:", last_batch_included_rag(state_after_tools_rag))
    print(
        "has_tool_named(clarify):",
        has_tool_named(batch, CLARIFICATION_TOOL_NAME),
    )

    section("after_tools — clarification tool just returned")
    state_after_tools_clarify: AgentState = {"messages": [ai_clarify, tool_clarify]}
    batch = preceding_ai_tool_calls(state_after_tools_clarify)
    print("preceding_ai_tool_calls:", batch)
    print(
        "has_tool_named(clarify):",
        has_tool_named(batch, CLARIFICATION_TOOL_NAME),
    )
    print("last_batch_included_rag:", last_batch_included_rag(state_after_tools_clarify))

    section("after_tools — web search (no RAG in batch)")
    state_after_tools_web: AgentState = {
        "messages": [
            ai_web,
            ToolMessage(content="snippet", tool_call_id="call_web"),
        ]
    }
    batch = preceding_ai_tool_calls(state_after_tools_web)
    print("preceding_ai_tool_calls:", batch)
    print("last_batch_included_rag:", last_batch_included_rag(state_after_tools_web))

# python -m agent.core.edges.tool_calls    #to run the example usage
# =======================example usage=======================