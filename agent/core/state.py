"""Agent graph state and metadata merge helpers."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, List, NotRequired, TypedDict

from agent.core.metadata.schema import AgentMetadata


class AgentState(TypedDict):
    """LangGraph state for agent.

    ``messages`` is append-only via ``add_messages``. Capabilities read/write
    ``metadata``; the LLM always sees the full message history.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    metadata: NotRequired[AgentMetadata]
    error: NotRequired[str]


def get_metadata(state: AgentState) -> AgentMetadata:
    """Return a shallow copy of metadata, or an empty dict if absent."""
    return cast("AgentMetadata", dict(state.get("metadata") or {}))


def merge_metadata(state: AgentState, patch: AgentMetadata) -> dict[str, AgentMetadata]:
    """Merge *patch* into current metadata and return a state update dict."""
    current = get_metadata(state)
    current.update(patch)
    return {"metadata": current}
