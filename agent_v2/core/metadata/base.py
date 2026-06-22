"""RAG invocation metadata shared across capabilities."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from agent_v2.core.constants import (
    DEFAULT_MAX_RAG_ATTEMPTS,
    DEFAULT_RAG_TOOL_NAME,
    DEFAULT_WEB_TOOL_NAME,
)

RAG_TOOL_NAME_KEY = "rag_tool_name"
RAG_ATTEMPT_KEY = "rag_attempt"
RAG_LAST_QUERY_KEY = "rag_last_query"
RAG_PROFILE_KEY = "rag_profile"
MAX_RAG_ATTEMPTS_KEY = "max_rag_attempts"

__all__ = [
    "DEFAULT_MAX_RAG_ATTEMPTS",
    "DEFAULT_RAG_TOOL_NAME",
    "DEFAULT_WEB_TOOL_NAME",
    "MAX_RAG_ATTEMPTS_KEY",
    "RAG_ATTEMPT_KEY",
    "RAG_LAST_QUERY_KEY",
    "RAG_PROFILE_KEY",
    "RAG_TOOL_NAME_KEY",
    "RagInvocationMeta",
]


class RagInvocationMeta(TypedDict, total=False):
    """Fields tracking RAG tool usage across the ReAct loop."""

    rag_tool_name: str
    rag_attempt: int
    rag_last_query: str
    rag_profile: dict[str, Any]
    max_rag_attempts: int
