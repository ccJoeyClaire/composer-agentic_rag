"""Retrieval-gate routing policy — block final answers when verdict fails."""

from __future__ import annotations

from agent.capabilities.retrieval_gate.config import RetrievalGateConfig
from agent.capabilities.retrieval_gate.metadata import (
    GATE_BLOCKED_TURNS_KEY,
    GATE_VERDICT_KEY,
)
from agent.config import AgentConfig
from agent.core.edges.tool_calls import last_ai_message
from agent.core.state import AgentState, get_metadata

_INSUFFICIENT_EVIDENCE_MARKERS = (
    "证据不足",
    "insufficient evidence",
    "无法基于已有",
    "cannot answer based on",
    "无法根据已有",
    "cannot answer based on the retrieved evidence",
)


def _resolved_gate_config(agent_config: AgentConfig) -> RetrievalGateConfig:
    return agent_config.retrieval_gate or RetrievalGateConfig()


def _blocked_turns(state: AgentState) -> int:
    raw = get_metadata(state).get(GATE_BLOCKED_TURNS_KEY)
    if raw is None:
        return 0
    return int(raw)


def _llm_declares_insufficient_evidence(state: AgentState) -> bool:
    """Heuristic: LLM opened the final answer with an honest-insufficiency phrase."""
    ai_msg = last_ai_message(state["messages"])
    if ai_msg is None or ai_msg.tool_calls:
        return False
    content = str(ai_msg.content or "").lower()
    return any(marker.lower() in content for marker in _INSUFFICIENT_EVIDENCE_MARKERS)


def gate_blocks_final_answer(state: AgentState, *, agent_config: AgentConfig) -> bool:
    """True when the LLM must not end the turn with a final answer.

    Applies only after retrieval was gated and the verdict is neither ``pass`` nor
    an internal scoring ``error`` (those are retried by the gate). Stops blocking
    after ``max_blocked_turns`` reject cycles or an explicit insufficient-evidence
    declaration in the latest AIMessage.
    """
    if not agent_config.enable_retrieval_gate:
        return False

    verdict = get_metadata(state).get(GATE_VERDICT_KEY)
    if verdict is None:
        return False
    if verdict in ("pass", "error"):
        return False

    gate_config = _resolved_gate_config(agent_config)
    if _blocked_turns(state) >= gate_config.max_blocked_turns:
        return False
    if _llm_declares_insufficient_evidence(state):
        return False
    return True
