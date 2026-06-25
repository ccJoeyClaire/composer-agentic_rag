"""Structured serialization for agent_v2 graph invoke results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage

from agent_v2.core.metadata.base import RAG_PROFILE_KEY
from agent_v2.core.metadata.schema import AgentMetadata
from agent_v2.core.state import AgentState
from agent_v2.capabilities.retrieval_gate.metadata import (
    GATE_ISSUES_KEY,
    GATE_PASSAGES_SUMMARY_KEY,
    GATE_VERDICT_KEY,
)
from agent_v2.pattern.common import RequestConfig


class MessageRecord(TypedDict, total=False):
    """One LangChain message as a JSON-friendly record."""

    type: str
    content: str
    tool_calls: list[dict[str, object]]


class AgentHighlights(TypedDict, total=False):
    """Frequently inspected metadata fields for demos and eval dumps."""

    gate_verdict: str | None
    gate_issues: list[str] | None
    rag_profile: dict[str, object] | None
    gate_passages_summary: str | None


class AgentRunRecord(TypedDict):
    """Full run record for JSON export (replaces ad-hoc build_run_record)."""

    pattern_id: str
    query: str
    collection: str
    profile_id: str
    enable_web_search: bool
    metadata: dict[str, object]
    highlights: AgentHighlights
    messages: list[MessageRecord]
    message_count: int
    final_message: MessageRecord | None


def _message_to_record(message: BaseMessage) -> MessageRecord:
    record: MessageRecord = {
        "type": message.__class__.__name__,
        "content": str(message.content),
    }
    if isinstance(message, AIMessage):
        record["tool_calls"] = list(getattr(message, "tool_calls", None) or [])
    return record


@dataclass
class OutputState:
    """Wrapper around a post-invoke ``AgentState`` with export helpers."""

    messages: list[BaseMessage]
    metadata: AgentMetadata
    query: str
    request_config: RequestConfig

    @classmethod
    def from_state(
        cls,
        state: AgentState | dict[str, object],
        *,
        query: str,
        request_config: RequestConfig,
    ) -> OutputState:
        """Build from ``graph.ainvoke`` return value."""
        messages = list(state.get("messages") or [])  # type: ignore[union-attr]
        raw_meta = state.get("metadata") or {}  # type: ignore[union-attr]
        metadata: AgentMetadata = dict(raw_meta)  # type: ignore[arg-type]
        return cls(
            messages=messages,
            metadata=metadata,
            query=query,
            request_config=request_config,
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable full payload."""
        return self.to_record_schema()

    def to_record_schema(self) -> AgentRunRecord:
        """Structured run record for JSON dumps."""
        meta = dict(self.metadata)
        last = self.messages[-1] if self.messages else None
        highlights: AgentHighlights = {
            "gate_verdict": meta.get(GATE_VERDICT_KEY),  # type: ignore[assignment]
            "gate_issues": meta.get(GATE_ISSUES_KEY),  # type: ignore[assignment]
            "rag_profile": meta.get(RAG_PROFILE_KEY),  # type: ignore[assignment]
            "gate_passages_summary": meta.get(GATE_PASSAGES_SUMMARY_KEY),  # type: ignore[assignment]
        }
        return AgentRunRecord(
            pattern_id=self.request_config.pattern_id,
            query=self.query,
            collection=self.request_config.collection,
            profile_id=self.request_config.profile_id,
            enable_web_search=self.request_config.enable_web_search,
            metadata=meta,
            highlights=highlights,
            messages=[_message_to_record(msg) for msg in self.messages],
            message_count=len(self.messages),
            final_message=_message_to_record(last) if last is not None else None,
        )

    def to_txt(self) -> str:
        """Pretty-print every message (LangChain ``pretty_repr`` per message)."""
        blocks = [message.pretty_repr() for message in self.messages]
        return "\n\n".join(blocks)

    def write_json(self, path: Path) -> None:
        """Write :meth:`to_record_schema` as indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_record_schema(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_doc(self, path: Path) -> None:
        """Write human-readable transcript (``.txt``)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_txt(), encoding="utf-8")
