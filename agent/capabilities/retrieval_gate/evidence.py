"""Extract user query and scorable passages from the latest tool batch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.capabilities.retrieval_gate.rag_context import split_rag_passages

EVIDENCE_SOURCE_RAG = "rag"
EVIDENCE_SOURCE_WEB = "web"

_WEB_ERROR_PREFIXES = ("Error:", "MCP ", "未配置", "未找到 MCP")


@dataclass(frozen=True)
class EvidenceBatch:
    """Passages from the latest RAG and/or web tool batch."""

    user_query: str
    passages: list[str]
    sources: tuple[str, ...]


def extract_user_query(messages: list[BaseMessage]) -> str:
    """Return the first human turn (original user question)."""
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.content or "")
    return ""


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


def _passages_from_web_items(items: list[Any]) -> list[str]:
    passages: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            passages.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        body = ""
        for key in ("content", "snippet", "raw_content", "text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                body = value.strip()
                break
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            passages.append(f"{title.strip()}\n{body}" if body else title.strip())
        elif body:
            passages.append(body)
    return passages


def split_web_passages(raw: str) -> list[str]:
    """Parse ``tavily_search`` / MCP text into passage strings for reranking."""
    text = raw.strip()
    if not text:
        return []
    if any(text.startswith(prefix) for prefix in _WEB_ERROR_PREFIXES):
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [text]

    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return _passages_from_web_items(results)
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return [answer.strip()]
        content = data.get("content")
        if isinstance(content, str) and content.strip():
            return [content.strip()]
        return []

    if isinstance(data, list):
        return _passages_from_web_items(data)

    return []


def extract_latest_evidence_batch(
    messages: list[BaseMessage],
    *,
    rag_tool_name: str,
    web_tool_name: str,
) -> EvidenceBatch | None:
    """Collect RAG + web passages from the latest tool batch; query is the user turn."""
    ai_msg, tool_messages = _latest_tool_batch(messages)
    if ai_msg is None or not ai_msg.tool_calls:
        return None

    call_by_id = {call["id"]: call for call in ai_msg.tool_calls}
    passages: list[str] = []
    sources: list[str] = []

    for tool_msg in tool_messages:
        call = call_by_id.get(tool_msg.tool_call_id)
        if call is None:
            continue
        name = call.get("name")
        raw = str(tool_msg.content or "")
        if name == rag_tool_name:
            rag_passages = split_rag_passages(raw)
            if rag_passages:
                passages.extend(rag_passages)
                if EVIDENCE_SOURCE_RAG not in sources:
                    sources.append(EVIDENCE_SOURCE_RAG)
        elif name == web_tool_name:
            web_passages = split_web_passages(raw)
            if web_passages:
                passages.extend(web_passages)
                if EVIDENCE_SOURCE_WEB not in sources:
                    sources.append(EVIDENCE_SOURCE_WEB)

    if not sources:
        return None

    return EvidenceBatch(
        user_query=extract_user_query(messages),
        passages=passages,
        sources=tuple(sources),
    )
