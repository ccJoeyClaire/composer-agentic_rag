"""Parse RAG tool outputs from AgentState messages.

Run (from repo root):
  python -m agent.reflection.parsers
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent.state import DEFAULT_RAG_TOOL_NAME

RAG_CHUNK_SEPARATOR = "\n\n---\n\n"


def split_rag_chunks(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [part.strip() for part in raw.split(RAG_CHUNK_SEPARATOR) if part.strip()]


def _latest_tool_batch(messages: list[BaseMessage]) -> tuple[AIMessage | None, list[ToolMessage]]:
    tool_messages: list[ToolMessage] = []
    idx = len(messages) - 1
    while idx >= 0 and isinstance(messages[idx], ToolMessage):
        tool_messages.insert(0, messages[idx])
        idx -= 1
    if not tool_messages:
        return None, []
    ai_msg = messages[idx] if idx >= 0 and isinstance(messages[idx], AIMessage) else None
    return ai_msg, tool_messages


def extract_rag_tool_results(
    messages: Iterable[BaseMessage],
    *,
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> list[dict[str, str]]:
    """Return RAG tool invocations from the latest tool batch."""
    msg_list = list(messages)
    ai_msg, tool_messages = _latest_tool_batch(msg_list)
    if not ai_msg or not ai_msg.tool_calls:
        return []

    call_by_id = {call["id"]: call for call in ai_msg.tool_calls}
    results: list[dict[str, str]] = []
    for tool_msg in tool_messages:
        call = call_by_id.get(tool_msg.tool_call_id)
        if call is None or call["name"] != rag_tool_name:
            continue
        query = str(call["args"].get("query", ""))
        results.append(
            {
                "tool_call_id": tool_msg.tool_call_id,
                "query": query,
                "raw": str(tool_msg.content or ""),
            }
        )
    return results


def _demo_main() -> None:
    """Offline smoke: extract latest RAG batch and split chunks."""
    from agent.state import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": DEFAULT_RAG_TOOL_NAME,
                    "args": {"query": "What is RAG?"},
                    "id": "tc_rag",
                },
                {
                    "name": DEFAULT_WEB_TOOL_NAME,
                    "args": {"query": "ignored"},
                    "id": "tc_web",
                },
            ],
        ),
        ToolMessage(content="alpha\n\n---\n\nbeta", tool_call_id="tc_rag"),
        ToolMessage(content="web result", tool_call_id="tc_web"),
    ]
    hits = extract_rag_tool_results(messages)
    print(f"extract_rag_tool_results: {len(hits)} RAG hit(s)")
    for hit in hits:
        chunks = split_rag_chunks(hit["raw"])
        print(f"  query={hit['query']!r} chunks={chunks}")


if __name__ == "__main__":
    _demo_main()
