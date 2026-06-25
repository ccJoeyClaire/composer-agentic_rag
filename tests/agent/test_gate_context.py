"""Tests for transient retrieval-gate LLM context."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.capabilities.retrieval_gate.gate_context import (
    build_gate_context_message,
    prepare_messages_for_llm,
)
from agent.capabilities.retrieval_gate.metadata import (
    GATE_ISSUES_KEY,
    GATE_PASSAGES_SUMMARY_KEY,
    GATE_VERDICT_KEY,
)


def test_build_gate_context_message_low_quality_includes_hard_block() -> None:
    message = build_gate_context_message(
        {
            GATE_VERDICT_KEY: "low_quality",
            GATE_PASSAGES_SUMMARY_KEY: "2 passage(s), max relevance=0.31",
            GATE_ISSUES_KEY: ["max relevance 0.31 below threshold 0.50"],
        }
    )
    assert message is not None
    assert "low_quality" in message.content
    assert "must NOT produce a final answer" in message.content


def test_build_gate_context_message_error_returns_none() -> None:
    assert (
        build_gate_context_message(
            {
                GATE_VERDICT_KEY: "error",
                GATE_ISSUES_KEY: ["scoring failed"],
            }
        )
        is None
    )


def test_prepare_messages_for_llm_inserts_after_existing_system() -> None:
    gate = SystemMessage(content="gate info")
    messages = [
        SystemMessage(content="base prompt"),
        HumanMessage(content="hello"),
    ]
    prepared = prepare_messages_for_llm(messages, gate_message=gate)
    assert len(prepared) == 3
    assert prepared[0].content == "base prompt"
    assert prepared[1].content == "gate info"
    assert isinstance(prepared[2], HumanMessage)
