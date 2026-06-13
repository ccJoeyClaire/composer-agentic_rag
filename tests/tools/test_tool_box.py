"""Unit tests for ToolBox."""

from __future__ import annotations

import pytest

from tools.registry import discover_packages, get_decorated_tools
from tools.tool_box import ToolBox


pytestmark = pytest.mark.unit


@pytest.fixture
def tool_box():
    return ToolBox()


def test_resolve_decorated_tool(tool_box):
    func = tool_box.resolve("tools.LocalTool.math_tool.integrate_function")
    assert func.__name__ == "integrate_function"


def test_autodiscover_registers_decorated_tools(tool_box):
    names = {s["function"]["name"] for s in tool_box.list_tools()}
    assert "integrate_function" in names
    assert "tavily_search" in names


@pytest.mark.asyncio
async def test_ainvoke_integrate(tool_box):
    result = await tool_box.ainvoke(
        "integrate_function",
        {"func_str": "x", "a": 0.0, "b": 1.0},
    )
    assert result.error is None
    assert result.output == pytest.approx(0.5)


def test_list_tools_schema(tool_box):
    schemas = tool_box.list_tools()
    names = {s["function"]["name"] for s in schemas}
    assert "integrate_function" in names


def test_local_tool_decorator_sets_metadata():
    discover_packages("tools.LocalTool.math_tool")
    info = get_decorated_tools()["integrate_function"]
    assert info.source == "local"
    assert info.tool_path == "tools.LocalTool.math_tool.integrate_function"


def test_autodiscover_can_be_disabled():
    box = ToolBox(autodiscover=False)
    assert box.list_tools() == []


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/tools/test_tool_box.py -v
# ================================================================================================================
