"""Tests for RAG tool output parsing and LLM-view preparation."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.core.tool_box import DEFAULT_RAG_TOOL_NAME
from agent.messages import (
    OMITTED_RAG_STUB,
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


def test_prune_rag_chunks_dedupes_by_chunk_id_and_keeps_first_in_order() -> None:
    chunks = [
        {"content": "a", "score": 0.9, "chunk_id": "x"},
        {"content": "a", "score": 0.2, "chunk_id": "x"},
        {"content": "b", "score": 0.5, "chunk_id": "y"},
    ]
    pruned = prune_rag_chunks(chunks, max_chunks=2)
    assert len(pruned) == 2
    assert pruned[0]["chunk_id"] == "x"
    assert pruned[0]["score"] == 0.9
    assert pruned[1]["chunk_id"] == "y"


def test_prepare_rag_context_keeps_recent_rounds_and_stubs_oldest_when_over_budget() -> None:
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
    prepared, pruned = prepare_rag_context_for_llm(
        original,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        max_chunks=2,
    )

    assert len(pruned) == 2
    assert pruned[0]["chunk_id"] == "c"
    assert pruned[1]["chunk_id"] == "b"
    assert prepared[2].content == OMITTED_RAG_STUB
    last = prepared[4]
    assert isinstance(last, ToolMessage)
    assert last.content == second
    assert original[4].content == second


def test_prepare_rag_context_keeps_multiple_rounds_within_window() -> None:
    first = json.dumps([{"content": "one", "chunk_id": "a"}])
    second = json.dumps([{"content": "two", "chunk_id": "b"}])
    original = _rag_messages([first, second])
    prepared, pruned = prepare_rag_context_for_llm(
        original,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        max_chunks=3,
    )

    assert [item["chunk_id"] for item in pruned] == ["a", "b"]
    assert prepared[2].content == first
    assert prepared[4].content == second


def test_prepare_rag_context_stubs_calls_outside_max_calls_window() -> None:
    contents = [
        json.dumps([{"content": f"chunk-{index}", "chunk_id": f"id-{index}"}])
        for index in range(4)
    ]
    original = _rag_messages(contents)
    prepared, pruned = prepare_rag_context_for_llm(
        original,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        max_chunks=9,
        max_calls=3,
    )

    assert prepared[2].content == OMITTED_RAG_STUB
    assert prepared[4].content == contents[1]
    assert prepared[6].content == contents[2]
    assert prepared[8].content == contents[3]
    assert [item["chunk_id"] for item in pruned] == ["id-1", "id-2", "id-3"]


def test_prepare_rag_context_does_not_mutate_state_messages() -> None:
    raw = json.dumps([{"content": "only", "score": 0.4, "chunk_id": "z"}])
    original = _rag_messages([raw])
    _, pruned = prepare_rag_context_for_llm(
        original,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        max_chunks=3,
    )
    assert len(pruned) == 1
    assert json.loads(original[2].content) == json.loads(raw)


def test_resolve_rag_context_max_chunks_defaults_to_three_times_top_k() -> None:
    assert resolve_rag_context_max_chunks(None) == 9


def test_resolve_rag_context_max_chunks_honors_override() -> None:
    assert resolve_rag_context_max_chunks(5) == 5
