"""AgentToolBox wrapper — filters tools by agent capability flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from tools.tool_box import ToolBox, ToolResult

from agent.core.tool_box.constants import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME
from agent.core.tool_box.rag_tool_policy import (
    rag_search_args_query_only,
    restrict_rag_search_tool_schema,
)


@dataclass
class AgentToolBox:
    """Thin wrapper around :class:`ToolBox` that applies agent tool policy.

    Filters web search when disabled, restricts ``RAG_search_tool`` to ``query``
    only unless the profile router is enabled, and injects extra tools (e.g.
    clarification) registered by capabilities.
    """

    inner: ToolBox
    enable_web_search: bool = True
    web_tool_name: str = DEFAULT_WEB_TOOL_NAME
    enable_rag_profile_router: bool = False
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME
    extra_tools: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def list_tools(self) -> List[Dict[str, Any]]:
        schemas = self.inner.list_tools()
        if not self.enable_web_search:
            schemas = [
                s
                for s in schemas
                if s.get("function", {}).get("name") != self.web_tool_name
            ]
        if not self.enable_rag_profile_router:
            schemas = [
                restrict_rag_search_tool_schema(
                    s,
                    rag_tool_name=self.rag_tool_name,
                )
                for s in schemas
            ]
        for name, func in self.extra_tools.items():
            from langchain_core.utils.function_calling import convert_to_openai_tool
            from langchain_core.tools import StructuredTool

            description = (func.__doc__ or "").strip()
            lc_tool = StructuredTool.from_function(func, name=name, description=description)
            schemas.append(convert_to_openai_tool(lc_tool))
        return schemas

    async def ainvoke(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name == self.rag_tool_name and not self.enable_rag_profile_router:
            args = rag_search_args_query_only(args)

        if name in self.extra_tools:
            func = self.extra_tools[name]
            try:
                import asyncio

                if asyncio.iscoroutinefunction(func):
                    output = await func(**args)
                else:
                    output = func(**args)
                return ToolResult(name=name, args=args, output=output, source="agent")
            except Exception as exc:
                return ToolResult(name=name, args=args, error=str(exc), source="agent")
        return await self.inner.ainvoke(name, args)
