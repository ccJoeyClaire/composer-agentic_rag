"""Strip a premature final answer blocked by the retrieval gate."""

from __future__ import annotations

from langchain_core.messages import AIMessage, RemoveMessage

from agent.core.state import AgentState


def strip_blocked_answer_node(state: AgentState) -> dict[str, object]:
    """Remove the latest AIMessage so the LLM can retry after a failed gate verdict."""
    if not state["messages"]:
        return {}

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or last.id is None:
        return {}

    return {"messages": [RemoveMessage(id=last.id)]}
