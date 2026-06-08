from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, TypeVar

ToolSource = Literal["local", "mcp"]

TOOL_INFO_ATTR = "__tool_info__"
_DECORATED: Dict[str, "ToolInfo"] = {}

DEFAULT_TOOL_PACKAGES = (
    "tools.LocalTool",
    "tools.MCPTool",
)

F = TypeVar("F", bound=Callable[..., object])


@dataclass
class ToolInfo:
    name: str
    source: ToolSource
    tool_path: str
    description: str = ""


def _register_decorated(
    func: F,
    *,
    source: ToolSource,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> F:
    tool_name = name or func.__name__
    tool_path = f"{func.__module__}.{func.__name__}"
    desc = description if description is not None else (func.__doc__ or "").strip()
    info = ToolInfo(
        name=tool_name,
        source=source,
        tool_path=tool_path,
        description=desc,
    )
    setattr(func, TOOL_INFO_ATTR, info)
    _DECORATED[tool_name] = info
    return func


def local_tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> F | Callable[[F], F]:
    """Mark a function as a local tool and record metadata for autodiscovery."""

    def decorator(fn: F) -> F:
        return _register_decorated(
            fn,
            source="local",
            name=name,
            description=description,
        )

    if func is not None:
        return decorator(func)
    return decorator


def mcp_tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> F | Callable[[F], F]:
    """Mark a function as an MCP-backed tool and record metadata for autodiscovery."""

    def decorator(fn: F) -> F:
        return _register_decorated(
            fn,
            source="mcp",
            name=name,
            description=description,
        )

    if func is not None:
        return decorator(func)
    return decorator


def get_decorated_tools() -> Dict[str, ToolInfo]:
    return dict(_DECORATED)


def clear_decorated_tools() -> None:
    _DECORATED.clear()


def discover_packages(*packages: str) -> None:
    """Import tool packages so @local_tool / @mcp_tool decorators run."""
    for package_name in packages:
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            continue
        for module in pkgutil.walk_packages(
            package.__path__,
            prefix=f"{package.__name__}.",
        ):
            module_name = module.name
            if module_name.endswith("._config") or module_name.endswith("._client"):
                continue
            importlib.import_module(module_name)
