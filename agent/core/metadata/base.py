"""RAG invocation metadata shared across capabilities."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from rag.serialize import ToolChunkRecord

from agent.core.tool_box import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME

RAG_TOOL_NAME_KEY = "rag_tool_name"
RAG_LAST_QUERY_KEY = "rag_last_query"
RAG_PROFILE_KEY = "rag_profile"
RETRIEVED_CONTEXT_KEY = "retrieved_context"

__all__ = [
    "DEFAULT_RAG_TOOL_NAME",
    "DEFAULT_WEB_TOOL_NAME",
    "RAG_LAST_QUERY_KEY",
    "RAG_PROFILE_KEY",
    "RAG_TOOL_NAME_KEY",
    "RETRIEVED_CONTEXT_KEY",
    "RagInvocationMeta",
]


class RagInvocationMeta(TypedDict, total=False):
    """Fields tracking RAG tool usage across the ReAct loop."""

    rag_tool_name: str
    rag_last_query: str
    rag_profile: dict[str, Any]
    retrieved_context: list[ToolChunkRecord]
