from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from tools.registry import DEFAULT_TOOL_PACKAGES, ToolInfo, discover_packages, get_decorated_tools


@dataclass
class ToolResult:
    name: str
    args: dict
    output: Any | None = None
    error: str | None = None
    source: str = "local"
    meta: dict = field(default_factory=dict)


class ToolBox:
    """Registry and runtime for decorator-discovered local and MCP tools."""

    def __init__(
        self,
        *,
        autodiscover: bool = True,
        packages: tuple[str, ...] | None = None,
    ) -> None:
        self._packages = packages or DEFAULT_TOOL_PACKAGES
        self._registry: Dict[str, ToolInfo] = {}
        self._cache: Dict[str, Callable[..., Any]] = {}
        if autodiscover:
            self._load_decorated_tools()

    def _load_decorated_tools(self) -> None:
        discover_packages(*self._packages)
        prefixes = tuple(f"{package}." for package in self._packages)
        self._registry = {
            name: info
            for name, info in get_decorated_tools().items()
            if info.tool_path.startswith(prefixes)
        }

    def resolve(self, tool_path: str) -> Callable[..., Any]:
        if tool_path in self._cache:
            return self._cache[tool_path]

        module_path, _, attr = tool_path.rpartition(".")
        if not module_path or not attr:
            raise ValueError(f"无效的 tool_path: {tool_path}")

        module = importlib.import_module(module_path)
        func = getattr(module, attr)
        if not callable(func):
            raise TypeError(f"{tool_path} 不是 callable")

        self._cache[tool_path] = func
        return func

    def _description_for(self, info: ToolInfo, func: Callable[..., Any]) -> str:
        return info.description or (func.__doc__ or "").strip()

    def _wrap_as_lc_tool(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
    ) -> BaseTool:
        if isinstance(func, BaseTool):
            return func
        kwargs: Dict[str, str] = {"name": name}
        if description:
            kwargs["description"] = description
        return StructuredTool.from_function(func, **kwargs)

    def list_tools(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for name, info in self._registry.items():
            func = self.resolve(info.tool_path)
            description = self._description_for(info, func)
            schemas.append(
                convert_to_openai_tool(
                    self._wrap_as_lc_tool(func, name, description)
                )
            )
        return schemas

    async def ainvoke(self, name: str, args: Dict[str, Any]) -> ToolResult:
        info = self._registry.get(name)
        if info is None:
            return ToolResult(
                name=name,
                args=args,
                error=f"未找到工具 {name}",
            )

        try:
            func = self.resolve(info.tool_path)
        except Exception as e:
            return ToolResult(
                name=name,
                args=args,
                error=f"无法加载工具 {info.tool_path}: {e}",
                source=info.source,
            )

        try:
            if asyncio.iscoroutinefunction(func):
                output = await func(**args)
            else:
                output = func(**args)
            return ToolResult(
                name=name,
                args=args,
                output=output,
                source=info.source,
            )
        except Exception as e:
            return ToolResult(
                name=name,
                args=args,
                error=str(e),
                source=info.source,
            )
