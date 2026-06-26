"""Retrieval-gate routing policy — block final answers when verdict fails."""

from __future__ import annotations

from agent.capabilities.retrieval_gate.metadata import GATE_VERDICT_KEY
from agent.config import AgentConfig
from agent.core.state import AgentState, get_metadata


def gate_blocks_final_answer(state: AgentState, *, agent_config: AgentConfig) -> bool:
    """True when the LLM must not end the turn with a final answer.

    Applies only after RAG retrieval was gated and the verdict is neither
    ``pass`` nor an internal scoring ``error`` (those are retried by the gate).
    """
    if not agent_config.enable_retrieval_gate:
        return False

    verdict = get_metadata(state).get(GATE_VERDICT_KEY)
    if verdict is None:
        return False
    if verdict in ("pass", "error"):
        return False
    return True
