"""Tests for RAG tool output parsing and LLM-view preparation."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.core.constants import DEFAULT_RAG_TOOL_NAME
from agent.messages import (
    MERGED_RAG_STUB,
    prepare_rag_context_for_llm,
    prune_rag_chunks,
    resolve_rag_context_max_chunks,
)
from rag.serialize import chunks_to_tool_json, parse_tool_chunks
from rag.base import Chunk

pytestmark = pytest.mark.unit


def _rag_messages(
    raw_contents: list[str],
    *,
    queries: list[str] | None = None,
) -> list:
    if queries is None:
        queries = [f"query-{index}" for index in range(len(raw_contents))]
    messages = [HumanMessage(content="question")]
    for index, (query, content) in enumerate(zip(queries, raw_contents, strict=True)):
        call_id = f"tc{index}"
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DEFAULT_RAG_TOOL_NAME,
                        "args": {"query": query},
                        "id": call_id,
                    }
                ],
            )
        )
        messages.append(ToolMessage(content=content, tool_call_id=call_id))
    return messages


def test_chunks_to_tool_json_includes_metadata_and_score() -> None:
    chunk = Chunk(
        content="hello world",
        metadata={
            "chunk_id": "doc.md::1",
            "source": "doc.md",
            "heading_path": "Intro",
            "start": 0,
            "end": 11,
        },
        score=0.87,
    )
    raw = chunks_to_tool_json([chunk])
    records = parse_tool_chunks(raw)
    assert len(records) == 1
    assert records[0]["content"] == "hello world"
    assert records[0]["score"] == 0.87
    assert records[0]["chunk_id"] == "doc.md::1"
    assert records[0]["heading_path"] == "Intro"


def test_prune_rag_chunks_dedupes_by_chunk_id_and_keeps_higher_score() -> None:
    chunks = [
        {"content": "a", "score": 0.2, "chunk_id": "x"},
        {"content": "a", "score": 0.9, "chunk_id": "x"},
        {"content": "b", "score": 0.5, "chunk_id": "y"},
    ]
    pruned = prune_rag_chunks(chunks, max_chunks=2)
    assert len(pruned) == 2
    assert pruned[0]["chunk_id"] == "x"
    assert pruned[0]["score"] == 0.9


def test_prepare_rag_context_merges_multiple_retrievals() -> None:
    first = json.dumps(
        [
            {"content": "low", "score": 0.1, "chunk_id": "a"},
            {"content": "mid", "score": 0.5, "chunk_id": "b"},
        ]
    )
    second = json.dumps(
        [
            {"content": "high", "score": 0.95, "chunk_id": "c"},
            {"content": "mid", "score": 0.6, "chunk_id": "b"},
        ]
    )
    original = _rag_messages([first, second])
    prepared = prepare_rag_context_for_llm(
        original,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        max_chunks=2,
    )

    assert prepared[2].content == MERGED_RAG_STUB
    last = prepared[4]
    assert isinstance(last, ToolMessage)
    assert "high" in last.content
    assert "mid" in last.content
    assert "low" not in last.content
    assert "score=0.9500" in last.content
    assert original[4].content == second


def test_prepare_rag_context_does_not_mutate_state_messages() -> None:
    raw = json.dumps([{"content": "only", "score": 0.4, "chunk_id": "z"}])
    original = _rag_messages([raw])
    prepare_rag_context_for_llm(
        original,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        max_chunks=3,
    )
    assert json.loads(original[2].content) == json.loads(raw)


def test_resolve_rag_context_max_chunks_defaults_to_three_times_top_k() -> None:
    assert resolve_rag_context_max_chunks(None) == 9


def test_resolve_rag_context_max_chunks_honors_override() -> None:
    assert resolve_rag_context_max_chunks(5) == 5
