"""Parse query and passages from the latest RAG tool batch."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

RAG_PASSAGE_SEPARATOR = "\n\n---\n\n"


def split_rag_passages(raw: str) -> list[str]:
    """Split ``RAG_search_tool`` output into individual passages."""
    if not raw or not raw.strip():
        return []
    return [
        part.strip()
        for part in raw.split(RAG_PASSAGE_SEPARATOR)
        if part.strip()
    ]


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
    """Return ``(query, raw_content)`` for the latest RAG tool in the last batch."""
    ai_msg, tool_messages = _latest_tool_batch(messages)
    if ai_msg is None or not ai_msg.tool_calls:
        return None

    call_by_id = {call["id"]: call for call in ai_msg.tool_calls}
    for tool_msg in reversed(tool_messages):
        call = call_by_id.get(tool_msg.tool_call_id)
        if call is None or call["name"] != rag_tool_name:
            continue
        query = str(call["args"].get("query", ""))
        return query, str(tool_msg.content or "")
    return None
