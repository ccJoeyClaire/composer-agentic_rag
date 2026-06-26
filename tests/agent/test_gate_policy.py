"""Tests for retrieval-gate routing policy."""

from __future__ import annotations

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph import END

from agent.capabilities.retrieval_gate.metadata import GATE_VERDICT_KEY
from agent.config import AgentConfig
from agent.core.edges.after_llm import route_after_llm
from agent.core.edges.gate_policy import gate_blocks_final_answer
from agent.core.edges.names import NodeName
from agent.core.nodes.gate_reject import strip_blocked_answer_node
from agent.core.state import AgentState
from unittest.mock import MagicMock


def _config(**kwargs: object) -> AgentConfig:
    return AgentConfig(llm=MagicMock(), enable_retrieval_gate=True, **kwargs)  # type: ignore[arg-type]


def test_gate_blocks_final_answer_only_for_failed_verdicts() -> None:
    state: AgentState = {
        "messages": [],
        "metadata": {GATE_VERDICT_KEY: "low_quality"},
    }
    assert gate_blocks_final_answer(state, agent_config=_config()) is True

    state["metadata"] = {GATE_VERDICT_KEY: "pass"}  # type: ignore[typeddict-item]
    assert gate_blocks_final_answer(state, agent_config=_config()) is False

    state["metadata"] = {GATE_VERDICT_KEY: "error"}  # type: ignore[typeddict-item]
    assert gate_blocks_final_answer(state, agent_config=_config()) is False


def test_route_after_llm_blocks_end_when_gate_fails() -> None:
    state: AgentState = {
        "messages": [
            AIMessage(content="premature answer", tool_calls=[], id="ai-final"),
        ],
        "metadata": {GATE_VERDICT_KEY: "low_quality"},
    }
    assert route_after_llm(state, agent_config=_config()) == NodeName.GATE_REJECT


def test_route_after_llm_allows_end_when_gate_passes() -> None:
    state: AgentState = {
        "messages": [AIMessage(content="answer", tool_calls=[])],
        "metadata": {GATE_VERDICT_KEY: "pass"},
    }
    assert route_after_llm(state, agent_config=_config()) == END


def test_strip_blocked_answer_node_removes_last_ai() -> None:
    state: AgentState = {
        "messages": [
            AIMessage(content="blocked", tool_calls=[], id="ai-1"),
        ]
    }
    patch = strip_blocked_answer_node(state)
    assert len(patch["messages"]) == 1
    assert isinstance(patch["messages"][0], RemoveMessage)
    assert patch["messages"][0].id == "ai-1"
