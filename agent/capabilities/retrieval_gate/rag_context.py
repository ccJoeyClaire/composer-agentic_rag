"""Parse query and passages from the latest RAG tool batch."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from rag.serialize import LEGACY_PASSAGE_SEPARATOR, parse_tool_chunks

RAG_PASSAGE_SEPARATOR = LEGACY_PASSAGE_SEPARATOR


def split_rag_passages(raw: str) -> list[str]:
    """Split ``RAG_search_tool`` output into individual passage texts."""
    chunks = parse_tool_chunks(raw)
    if chunks:
        return [chunk["content"] for chunk in chunks if chunk.get("content")]
    return []


def _latest_tool_batch(
    messages: list[BaseMessage],
) -> tuple[AIMessage | None, list[ToolMessage]]:
    tool_messages: list[ToolMessage] = []
    index = len(messages) - 1
    while index >= 0 and isinstance(messages[index], ToolMessage):
        tool_messages.insert(0, messages[index])
        index -= 1
    if not tool_messages:
        return None, []
    ai_msg = messages[index] if index >= 0 and isinstance(messages[index], AIMessage) else None
    return ai_msg, tool_messages


def extract_latest_rag_context(
    messages: list[BaseMessage],
    *,
    rag_tool_name: str,
) -> tuple[str, str] | None:
    """Return ``(user_query, raw_rag_content)`` for the latest RAG tool in the last batch."""
    from agent.capabilities.retrieval_gate.evidence import (
        EVIDENCE_SOURCE_RAG,
        extract_latest_evidence_batch,
        extract_user_query,
    )

    batch = extract_latest_evidence_batch(
        messages,
        rag_tool_name=rag_tool_name,
        web_tool_name="__unused__",
    )
    if batch is None or EVIDENCE_SOURCE_RAG not in batch.sources:
        return None

    ai_msg, tool_messages = _latest_tool_batch(messages)
    if ai_msg is None:
        return None
    call_by_id = {call["id"]: call for call in ai_msg.tool_calls}
    for tool_msg in reversed(tool_messages):
        call = call_by_id.get(tool_msg.tool_call_id)
        if call is None or call["name"] != rag_tool_name:
            continue
        return extract_user_query(messages), str(tool_msg.content or "")
    return None
