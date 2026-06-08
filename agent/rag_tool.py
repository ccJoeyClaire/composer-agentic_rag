"""Thin ToolBox wrapper for agent-layer RAG re-invocation."""

from __future__ import annotations

from agent.metadata_schema import DEFAULT_RAG_TOOL_NAME
from tools.tool_box import ToolBox, ToolResult


async def invoke_rag_tool(
    tool_box: ToolBox,
    *,
    query: str,
    tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> ToolResult:
    """Invoke RAG search through ToolBox (never import RAGRetriever here)."""
    return await tool_box.ainvoke(tool_name, {"query": query})
