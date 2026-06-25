"""AgentInferArm: run agent_v2 graph for one query.

Covers all four agent patterns (react / crag / self_rag / crag_self_rag).
``retrieved_context`` is extracted from ToolMessage content (blueprint §4.2 A).
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent_v2.output import OutputState
from agent_v2.pattern.common import RequestConfig, build_graph

from eval.infer.base import BaseInferArm
from eval.run_config import EvalRunConfig
from eval.types import ContextChunk, InferResult

# ToolMessage content is formatted as:
#   Document: ...\nSource: <source>\nSection: ...\nKeywords: ...\n\n<body>
# Multiple chunks are separated by "\n\n---\n\n".
_CHUNK_SEPARATOR = "\n\n---\n\n"
_SOURCE_LINE = re.compile(r"^Source:\s*(.+)$", re.MULTILINE)


def _parse_tool_message(content: str) -> list[ContextChunk]:
    """Extract deduplicated context chunks from one ToolMessage body.

    Each chunk block is split on ``---`` separators.  ``doc_id`` is taken
    from the ``Source:`` header line; ``text`` is the body after the header
    block (no contextual header prefix per blueprint §4.2).

    Args:
        content: Raw ToolMessage content string.

    Returns:
        Parsed chunks; duplicates (same doc_id + text) are removed.
    """
    seen: set[tuple[str, str]] = set()
    chunks: list[ContextChunk] = []

    for block in content.split(_CHUNK_SEPARATOR):
        block = block.strip()
        if not block:
            continue

        source_match = _SOURCE_LINE.search(block)
        doc_id = source_match.group(1).strip() if source_match else ""

        # Body starts after the last header line (Keywords: ...).
        body_start = block.find("\n\n")
        text = block[body_start:].strip() if body_start != -1 else block

        key = (doc_id, text)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(ContextChunk(doc_id=doc_id, text=text))

    return chunks


def _extract_context_from_messages(messages: list[BaseMessage]) -> list[ContextChunk]:
    """Merge and deduplicate context from all ToolMessages in the trace.

    Args:
        messages: Full LangChain message list from graph invoke.

    Returns:
        Deduplicated context chunks in encounter order.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[ContextChunk] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for chunk in _parse_tool_message(str(msg.content)):
            key = (chunk["doc_id"], chunk["text"])
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
    return merged


def _final_response(messages: list[BaseMessage]) -> str:
    """Return the content of the last AIMessage without tool calls."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return str(msg.content)
    return ""


class AgentInferArm(BaseInferArm):
    """Run one agent_v2 pattern for a single query.

    Args:
        config: Run config supplying profile_id, collection, and arm (pattern).
    """

    def __init__(self, config: EvalRunConfig) -> None:
        super().__init__(config)
        request_config = RequestConfig(
            pattern_id=config.arm,
            collection=config.collection,
            profile_id=config.profile_id,
        )
        self._request_config = request_config
        self._graph = build_graph(request_config)

    async def arun(self, query_id: str, query: str) -> InferResult:
        """Invoke the agent graph and extract response + context.

        Args:
            query_id: Stable gold record identifier.
            query:    Natural-language question string.

        Returns:
            :class:`InferResult` with final AIMessage content and ToolMessage context.
        """
        from langchain_core.messages import HumanMessage

        raw = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=query)], "metadata": {}}
        )
        output = OutputState.from_state(
            raw,
            query=query,
            request_config=self._request_config,
        )

        return InferResult(
            query_id=query_id,
            query=query,
            response=_final_response(output.messages),
            retrieved_context=_extract_context_from_messages(output.messages),
            arm=self._config.arm,
            profile_id=self._config.profile_id,
        )
