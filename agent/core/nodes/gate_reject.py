"""Strip a premature final answer blocked by the retrieval gate."""

from __future__ import annotations

from langchain_core.messages import AIMessage, RemoveMessage

from agent.capabilities.retrieval_gate.metadata import GATE_BLOCKED_TURNS_KEY
from agent.core.state import AgentState, get_metadata, merge_metadata


def strip_blocked_answer_node(state: AgentState) -> dict[str, object]:
    """Remove the latest AIMessage so the LLM can retry after a failed gate verdict."""
    if not state["messages"]:
        return {}

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or last.id is None:
        return {}

    blocked = _blocked_turns(state) + 1
    patch: dict[str, object] = {"messages": [RemoveMessage(id=last.id)]}
    patch.update(merge_metadata(state, {GATE_BLOCKED_TURNS_KEY: blocked}))
    return patch


def _blocked_turns(state: AgentState) -> int:
    raw = get_metadata(state).get(GATE_BLOCKED_TURNS_KEY)
    if raw is None:
        return 0
    return int(raw)
