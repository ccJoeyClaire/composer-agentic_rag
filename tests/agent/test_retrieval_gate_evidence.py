"""Tests for retrieval_gate evidence extraction."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.capabilities.retrieval_gate.evidence import (
    EVIDENCE_SOURCE_RAG,
    EVIDENCE_SOURCE_WEB,
    extract_latest_evidence_batch,
    extract_user_query,
    split_web_passages,
)
from agent.capabilities.retrieval_gate.rag_context import (
    RAG_PASSAGE_SEPARATOR,
    extract_latest_rag_context,
)
from agent.core.tool_box import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME

pytestmark = pytest.mark.unit


def test_extract_user_query() -> None:
    messages = [
        HumanMessage(content="original question"),
        AIMessage(content="thinking"),
    ]
    assert extract_user_query(messages) == "original question"


def test_split_web_passages_json_results() -> None:
    raw = json.dumps(
        {
            "results": [
                {"title": "Titanic", "content": "James Cameron directed Titanic."},
                {"title": "Avatar", "content": "James Cameron also directed Avatar."},
            ]
        }
    )
    passages = split_web_passages(raw)
    assert len(passages) == 2
    assert "Titanic" in passages[0]
    assert "Avatar" in passages[1]


def test_split_web_passages_error_returns_empty() -> None:
    assert split_web_passages("Error: 未配置 TAVILY_MCP_URL") == []


def test_extract_latest_evidence_batch_merges_rag_and_web() -> None:
    user_query = "Who directed Titanic?"
    rag_raw = f"kb hit{RAG_PASSAGE_SEPARATOR}second hit"
    web_raw = json.dumps(
        {"results": [{"title": "Web", "content": "James Cameron directed Titanic."}]}
    )
    messages = [
        HumanMessage(content=user_query),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": DEFAULT_RAG_TOOL_NAME,
                    "args": {"query": "Titanic director"},
                    "id": "tc_rag",
                },
                {
                    "name": DEFAULT_WEB_TOOL_NAME,
                    "args": {"query": "Titanic director films"},
                    "id": "tc_web",
                },
            ],
        ),
        ToolMessage(content=rag_raw, tool_call_id="tc_rag"),
        ToolMessage(content=web_raw, tool_call_id="tc_web"),
    ]

    batch = extract_latest_evidence_batch(
        messages,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        web_tool_name=DEFAULT_WEB_TOOL_NAME,
    )
    assert batch is not None
    assert batch.user_query == user_query
    assert batch.sources == (EVIDENCE_SOURCE_RAG, EVIDENCE_SOURCE_WEB)
    assert batch.passages == ["kb hit", "second hit", "Web\nJames Cameron directed Titanic."]


def test_extract_latest_evidence_batch_web_only() -> None:
    web_raw = json.dumps(
        {"results": [{"title": "News", "content": "Live update about the event."}]}
    )
    messages = [
        HumanMessage(content="What happened today?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": DEFAULT_WEB_TOOL_NAME,
                    "args": {"query": "today news"},
                    "id": "tc_web",
                }
            ],
        ),
        ToolMessage(content=web_raw, tool_call_id="tc_web"),
    ]

    batch = extract_latest_evidence_batch(
        messages,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        web_tool_name=DEFAULT_WEB_TOOL_NAME,
    )
    assert batch is not None
    assert batch.sources == (EVIDENCE_SOURCE_WEB,)
    assert len(batch.passages) == 1


def test_extract_latest_rag_context_uses_user_query() -> None:
    messages = [
        HumanMessage(content="user question"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": DEFAULT_RAG_TOOL_NAME,
                    "args": {"query": "rewritten"},
                    "id": "tc1",
                }
            ],
        ),
        ToolMessage(content="chunk one", tool_call_id="tc1"),
    ]
    assert extract_latest_rag_context(messages, rag_tool_name=DEFAULT_RAG_TOOL_NAME) == (
        "user question",
        "chunk one",
    )
