"""Tests for retrieval-gate routing policy."""

from __future__ import annotations

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph import END
from unittest.mock import MagicMock

from agent.capabilities.retrieval_gate.config import RetrievalGateConfig
from agent.capabilities.retrieval_gate.metadata import (
    GATE_BLOCKED_TURNS_KEY,
    GATE_VERDICT_KEY,
)
from agent.config import AgentConfig
from agent.core.edges.after_llm import route_after_llm
from agent.core.edges.after_tools import route_after_tools
from agent.core.edges.gate_policy import gate_blocks_final_answer
from agent.core.edges.names import NodeName
from agent.core.edges.tool_calls import last_batch_included_scorable_evidence
from agent.core.nodes.gate_reject import strip_blocked_answer_node
from agent.core.state import AgentState
from agent.core.tool_box import DEFAULT_WEB_TOOL_NAME
from langchain_core.messages import HumanMessage, ToolMessage


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


def test_gate_blocks_stops_after_max_blocked_turns() -> None:
    state: AgentState = {
        "messages": [],
        "metadata": {
            GATE_VERDICT_KEY: "low_quality",
            GATE_BLOCKED_TURNS_KEY: 5,
        },
    }
    config = _config(retrieval_gate=RetrievalGateConfig(max_blocked_turns=5))
    assert gate_blocks_final_answer(state, agent_config=config) is False


def test_gate_blocks_allows_insufficient_evidence_answer() -> None:
    state: AgentState = {
        "messages": [
            AIMessage(
                content="```\nAction:\n证据不足，无法根据已有检索结果作答。\n```",
                tool_calls=[],
            )
        ],
        "metadata": {GATE_VERDICT_KEY: "low_quality"},
    }
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


def test_route_after_llm_allows_end_when_blocked_turns_exceeded() -> None:
    state: AgentState = {
        "messages": [AIMessage(content="best effort answer", tool_calls=[])],
        "metadata": {
            GATE_VERDICT_KEY: "low_quality",
            GATE_BLOCKED_TURNS_KEY: 5,
        },
    }
    config = _config(retrieval_gate=RetrievalGateConfig(max_blocked_turns=5))
    assert route_after_llm(state, agent_config=config) == END


def test_strip_blocked_answer_node_removes_last_ai_and_increments_blocked_turns() -> None:
    state: AgentState = {
        "messages": [
            AIMessage(content="blocked", tool_calls=[], id="ai-1"),
        ],
        "metadata": {GATE_BLOCKED_TURNS_KEY: 1},
    }
    patch = strip_blocked_answer_node(state)
    assert len(patch["messages"]) == 1
    assert isinstance(patch["messages"][0], RemoveMessage)
    assert patch["messages"][0].id == "ai-1"
    assert patch["metadata"][GATE_BLOCKED_TURNS_KEY] == 2  # type: ignore[index]


def test_route_after_tools_routes_web_batch_to_gate() -> None:
    state: AgentState = {
        "messages": [
            HumanMessage(content="Who directed Titanic?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DEFAULT_WEB_TOOL_NAME,
                        "args": {"query": "Titanic director"},
                        "id": "tc_web",
                    }
                ],
            ),
            ToolMessage(content='{"results": []}', tool_call_id="tc_web"),
        ],
        "metadata": {},
    }
    config = _config(enable_web_search=True)
    assert last_batch_included_scorable_evidence(state, agent_config=config) is True
    assert route_after_tools(state, agent_config=config) == NodeName.RETRIEVAL_GATE


def test_route_after_tools_skips_gate_when_web_disabled() -> None:
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DEFAULT_WEB_TOOL_NAME,
                        "args": {"query": "q"},
                        "id": "tc_web",
                    }
                ],
            ),
            ToolMessage(content="web", tool_call_id="tc_web"),
        ],
        "metadata": {},
    }
    config = _config(enable_web_search=False)
    assert last_batch_included_scorable_evidence(state, agent_config=config) is False
    assert route_after_tools(state, agent_config=config) == NodeName.LLM
