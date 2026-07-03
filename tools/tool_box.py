from __future__ import annotations

import asyncio
import copy
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from tools.context import INJECTED_CONTEXT_PARAM, ToolContextBundle
from tools.registry import DEFAULT_TOOL_PACKAGES, ToolInfo, discover_packages, get_registered_tools


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
        context: ToolContextBundle | None = None,
        autodiscover: bool = True,
        packages: tuple[str, ...] | None = None,
    ) -> None:
        self._context = context or ToolContextBundle()
        self._packages = packages or DEFAULT_TOOL_PACKAGES
        self._registry: Dict[str, ToolInfo] = {}
        self._cache: Dict[str, Callable[..., Any]] = {}
        if autodiscover:
            self._load_registered_tools()

    @property
    def context(self) -> ToolContextBundle:
        return self._context

    def _load_registered_tools(self) -> None:
        discover_packages(*self._packages)
        prefixes = tuple(f"{package}." for package in self._packages)
        self._registry = {
            name: info
            for name, info in get_registered_tools().items()
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

    def _strip_injected_params(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        props = schema.get("function", {}).get("parameters", {}).get("properties")
        if not props or INJECTED_CONTEXT_PARAM not in props:
            return schema

        filtered = copy.deepcopy(schema)
        params = filtered["function"]["parameters"]
        params["properties"] = {
            key: value
            for key, value in params["properties"].items()
            if key != INJECTED_CONTEXT_PARAM
        }
        required = params.get("required") or []
        params["required"] = [key for key in required if key != INJECTED_CONTEXT_PARAM]
        return filtered

    def list_tools(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for name, info in self._registry.items():
            func = self.resolve(info.tool_path)
            description = self._description_for(info, func)
            schema = convert_to_openai_tool(
                self._wrap_as_lc_tool(func, name, description)
            )
            schemas.append(self._strip_injected_params(schema))
        return schemas

    async def ainvoke(self, name: str, args: Dict[str, Any]) -> ToolResult:
        info = self._registry.get(name)
        if info is None:
            return ToolResult(
                name=name,
                args=args,
                error=f"未找到工具 {name}",
            )

        missing = [key for key in info.context_keys if not self._context.has(key)]
        if missing:
            return ToolResult(
                name=name,
                args=args,
                error=f"Missing tool context: {', '.join(missing)}",
                source=info.source,
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

        invoke_args = dict(args)
        if info.context_keys:
            invoke_args[INJECTED_CONTEXT_PARAM] = self._context.view(info.context_keys)

        try:
            if asyncio.iscoroutinefunction(func):
                output = await func(**invoke_args)
            else:
                output = func(**invoke_args)
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

    async def aclose(self) -> None:
        await self._context.aclose()


def _main() -> None:
    """List or invoke registered tools.

    Run (from repo root):
      python -m tools.tool_box list
      python -m tools.tool_box invoke integrate_function --args '{"func_str":"x**2","a":0,"b":1}'
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="List or invoke ToolBox tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Print registered tool names and schemas")

    invoke_p = sub.add_parser("invoke", help="Call a tool by name")
    invoke_p.add_argument("name", help="Registered tool name")
    invoke_p.add_argument(
        "--args",
        default="{}",
        help='JSON object of tool arguments, e.g. \'{"func_str":"x**2","a":0,"b":1}\'',
    )

    args = parser.parse_args()
    box = ToolBox()

    if args.command == "list":
        for schema in box.list_tools():
            fn = schema.get("function", {})
            print(f"{fn.get('name')}: {fn.get('description', '')[:80]}")
        return

    try:
        tool_args = json.loads(args.args)
    except json.JSONDecodeError as exc:
        print(f"Invalid --args JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(box.ainvoke(args.name, tool_args))
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)
    print(result.output)


if __name__ == "__main__":
    _main()
