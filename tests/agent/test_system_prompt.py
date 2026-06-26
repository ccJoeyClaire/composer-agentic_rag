"""Tests for system prompt loading and graph seed node."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agent.core.nodes.seed import seed_system_prompt_node
from agent.core.state import AgentState
from agent.prompt.load import get_system_prompt, load_system_prompts

pytestmark = pytest.mark.unit


def test_load_system_prompts_includes_default() -> None:
    prompts = load_system_prompts()
    assert "default" in prompts
    assert "Think:" in prompts["default"]
    assert "Action:" in prompts["default"]
    assert "en" in prompts
    assert "Think:" in prompts["en"]


def test_get_system_prompt_unknown_key_raises() -> None:
    with pytest.raises(KeyError, match="unknown"):
        get_system_prompt("unknown")


def test_seed_node_injects_prompt_for_human_only_input() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "metadata": {},
    }
    patch = seed_system_prompt_node(state, system_prompt_key="default")
    messages = patch["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert "Think:" in messages[0].content
    assert isinstance(messages[1], HumanMessage)


def test_seed_node_ignores_caller_system_message() -> None:
    state: AgentState = {
        "messages": [
            SystemMessage(content="caller must not control this"),
            HumanMessage(content="hello"),
        ],
        "metadata": {},
    }
    patch = seed_system_prompt_node(state, system_prompt_key="default")
    messages = patch["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == get_system_prompt("default")
    assert "caller must not" not in messages[0].content


def test_seed_node_idempotent_when_already_seeded() -> None:
    prompt = get_system_prompt("default")
    state: AgentState = {
        "messages": [
            SystemMessage(content=prompt),
            HumanMessage(content="hello"),
        ],
        "metadata": {},
    }
    assert seed_system_prompt_node(state, system_prompt_key="default") == {}
