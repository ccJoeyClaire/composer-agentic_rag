"""Tests for agent OutputState serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.output import OutputState
from agent.pattern.common import RequestConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def request_config() -> RequestConfig:
    return RequestConfig(
        pattern_id="self_rag",
        collection="demo_collection",
        profile_id="baseline",
        enable_web_search=False,
    )


def test_from_state_to_record_schema(request_config: RequestConfig) -> None:
    query = "What is RAG?"
    raw = {
        "messages": [
            HumanMessage(content=query),
            AIMessage(content="Searching.", tool_calls=[]),
            ToolMessage(content="passage one", tool_call_id="c1"),
            AIMessage(content="Final answer.", tool_calls=[]),
        ],
        "metadata": {"profile_validated": True},
    }
    output = OutputState.from_state(raw, query=query, request_config=request_config)
    record = output.to_record_schema()

    assert record["pattern_id"] == "self_rag"
    assert record["query"] == query
    assert record["collection"] == "demo_collection"
    assert record["message_count"] == 4
    assert record["final_message"]["content"] == "Final answer."
    tool_record = record["messages"][2]
    assert tool_record["type"] == "ToolMessage"
    assert tool_record["tool_call_id"] == "c1"


def test_to_txt_uses_pretty_repr(request_config: RequestConfig) -> None:
    output = OutputState.from_state(
        {"messages": [HumanMessage(content="Hi")], "metadata": {}},
        query="Hi",
        request_config=request_config,
    )
    text = output.to_txt()
    assert "Human Message" in text
    assert "Hi" in text


def test_write_json_and_doc(
    request_config: RequestConfig,
    tmp_path: Path,
) -> None:
    output = OutputState.from_state(
        {
            "messages": [HumanMessage(content="Q"), AIMessage(content="A")],
            "metadata": {},
        },
        query="Q",
        request_config=request_config,
    )
    json_path = tmp_path / "run.json"
    doc_path = tmp_path / "run.txt"
    output.write_json(json_path)
    output.write_doc(doc_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["query"] == "Q"
    assert doc_path.read_text(encoding="utf-8") == output.to_txt()
