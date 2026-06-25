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
MERGED_RAG_STUB = "(retrieval context consolidated in the latest RAG result below)"


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
    """Deduplicate by ``chunk_id`` (or content) and keep top *max_chunks* by score."""
    best_by_key: dict[str, ToolChunkRecord] = {}
    for chunk in chunks:
        key = _chunk_dedupe_key(chunk)
        existing = best_by_key.get(key)
        if existing is None or chunk.get("score", 0.0) > existing.get("score", 0.0):
            best_by_key[key] = chunk
    ranked = sorted(
        best_by_key.values(),
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )
    return ranked[:max_chunks]


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
) -> list[BaseMessage]:
    """Return a message copy with merged, score-pruned RAG context for the LLM.

    ``AgentState`` is unchanged; only the view passed to the LLM is rewritten.
    """
    call_names = _tool_call_names(messages)
    rag_indices: list[int] = []
    all_chunks: list[ToolChunkRecord] = []

    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        if call_names.get(message.tool_call_id) != rag_tool_name:
            continue
        rag_indices.append(index)
        all_chunks.extend(parse_tool_chunks(str(message.content or "")))

    if not rag_indices:
        return list(messages)

    pruned = prune_rag_chunks(all_chunks, max_chunks=max_chunks)
    formatted = format_rag_chunks_for_llm(pruned)
    last_index = rag_indices[-1]
    _, notes_suffix = split_tool_body_and_notes(
        str(messages[last_index].content or "")
    )

    result: list[BaseMessage] = list(messages)
    for index in rag_indices[:-1]:
        message = result[index]
        assert isinstance(message, ToolMessage)
        result[index] = message.model_copy(update={"content": MERGED_RAG_STUB})

    last_message = result[last_index]
    assert isinstance(last_message, ToolMessage)
    last_content = formatted + notes_suffix if formatted else notes_suffix
    result[last_index] = last_message.model_copy(update={"content": last_content})
    return result


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
