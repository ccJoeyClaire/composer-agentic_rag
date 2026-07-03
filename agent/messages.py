"""LangChain message adapters and LLM-view preparation for agent."""

from __future__ import annotations

import json
from typing import Any, List, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from rag.config import get_rag_config
from rag.serialize import (
    LEGACY_PASSAGE_SEPARATOR,
    ToolChunkRecord,
    parse_tool_chunks,
    split_tool_body_and_notes,
)

RAG_CONTEXT_TOP_K_MULTIPLIER = 3
RAG_CONTEXT_MAX_CALLS_DEFAULT = 3
OMITTED_RAG_STUB = "(earlier RAG retrieval omitted from LLM context)"
MERGED_RAG_STUB = OMITTED_RAG_STUB


def resolve_rag_context_max_chunks(configured: int | None) -> int:
    """Return configured cap or ``3 * retriever.top_k`` from ``arg_config.yaml``."""
    if configured is not None:
        return max(1, configured)
    return RAG_CONTEXT_TOP_K_MULTIPLIER * get_rag_config().retriever.top_k


def _chunk_dedupe_key(chunk: ToolChunkRecord) -> str:
    chunk_id = chunk.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id.strip():
        return f"id:{chunk_id}"
    return f"content:{chunk.get('content', '').strip()}"


def prune_rag_chunks(
    chunks: list[ToolChunkRecord],
    *,
    max_chunks: int,
) -> list[ToolChunkRecord]:
    """Keep the first *max_chunks* unique chunks in retrieval order."""
    seen: set[str] = set()
    kept: list[ToolChunkRecord] = []
    for chunk in chunks:
        key = _chunk_dedupe_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        kept.append(chunk)
        if len(kept) >= max_chunks:
            break
    return kept


def _format_chunk_header(chunk: ToolChunkRecord) -> str:
    parts: list[str] = []
    score = chunk.get("score")
    if score is not None:
        parts.append(f"score={score:.4f}")
    for key, label in (
        ("source", "source"),
        ("chunk_id", "chunk_id"),
        ("heading_path", "heading"),
    ):
        value = chunk.get(key)  # type: ignore[arg-type]
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}={value}")
    start = chunk.get("start")
    end = chunk.get("end")
    if isinstance(start, int) and isinstance(end, int):
        parts.append(f"chars={start}-{end}")
    boundary = chunk.get("boundary_reason")
    if isinstance(boundary, str) and boundary.strip():
        parts.append(f"boundary={boundary}")
    if not parts:
        return "[chunk]"
    return "[" + " | ".join(parts) + "]"


def format_rag_chunks_for_llm(chunks: list[ToolChunkRecord]) -> str:
    """Render pruned chunks as LLM-readable text (not persisted in state)."""
    if not chunks:
        return ""
    blocks = [
        f"{_format_chunk_header(chunk)}\n{chunk.get('content', '')}"
        for chunk in chunks
    ]
    return LEGACY_PASSAGE_SEPARATOR.join(blocks)


def _tool_call_names(messages: list[BaseMessage]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        for call in message.tool_calls:
            call_id = call.get("id")
            name = call.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


def prepare_rag_context_for_llm(
    messages: list[BaseMessage],
    *,
    rag_tool_name: str,
    max_chunks: int,
    max_calls: int = RAG_CONTEXT_MAX_CALLS_DEFAULT,
) -> tuple[list[BaseMessage], list[ToolChunkRecord]]:
    """Return LLM view and RAG chunks shown to the model.

    Keeps the newest *max_calls* RAG tool results as-is; stubs older rounds.
    If the window still exceeds *max_chunks*, drops the earliest kept round;
    a single overflowing round is truncated in retrieval order.

    The returned message list is a copy; *messages* in ``AgentState`` are unchanged.
    """
    call_names = _tool_call_names(messages)
    rag_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage)
        and call_names.get(message.tool_call_id) == rag_tool_name
    ]
    if not rag_indices:
        return list(messages), []

    def _chunks_at(index: int) -> list[ToolChunkRecord]:
        return parse_tool_chunks(str(messages[index].content or ""))

    kept = rag_indices[-max(1, max_calls) :]
    while len(kept) > 1 and sum(len(_chunks_at(i)) for i in kept) > max_chunks:
        kept = kept[1:]
    kept_set = set(kept)

    result = list(messages)
    retrieved: list[ToolChunkRecord] = []
    budget = max_chunks

    for index in rag_indices:
        message = result[index]
        assert isinstance(message, ToolMessage)
        if index not in kept_set:
            result[index] = message.model_copy(update={"content": OMITTED_RAG_STUB})
            continue

        raw = str(messages[index].content or "")
        chunks = _chunks_at(index)
        if len(chunks) > budget:
            chunks = prune_rag_chunks(chunks, max_chunks=budget)
            body, notes = split_tool_body_and_notes(raw)
            content = (
                json.dumps(chunks, ensure_ascii=False) + notes
                if body.strip().startswith("[")
                else format_rag_chunks_for_llm(chunks) + notes
            )
            result[index] = message.model_copy(update={"content": content})
        retrieved.extend(chunks)
        budget -= len(chunks)

    return result, retrieved


def messages_to_openai(messages: List[Union[BaseMessage, dict]]) -> List[dict]:
    """Convert LangChain messages to OpenAI chat-completions format."""
    result: List[dict] = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
            continue
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")
    return result


def openai_response_to_ai_message(response: Any) -> AIMessage:
    """Parse an OpenAI-style LLM response into an AIMessage."""
    tool_calls = None
    if getattr(response, "tool_calls", None):
        tool_calls = [
            {
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "id": tc.id,
            }
            for tc in response.tool_calls
        ]
    return AIMessage(content=response.content or "", tool_calls=tool_calls or [])
