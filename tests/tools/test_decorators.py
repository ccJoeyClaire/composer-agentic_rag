"""Verify @local_tool on tools.LocalTool.math_tool.integrate_function."""

from __future__ import annotations

import pytest

from tools.LocalTool import math_tool
from tools.registry import TOOL_INFO_ATTR
from tools.tool_box import ToolBox

pytestmark = pytest.mark.unit

MATH_TOOL_PACKAGE = ("tools.LocalTool.math_tool",)


@pytest.fixture
def math_tool_box():
    """Only load math_tool — avoid pulling in MCPTool / other LocalTool modules."""
    return ToolBox(autodiscover=True, packages=MATH_TOOL_PACKAGE)


def test_local_tool_attaches_tool_info():
    info = getattr(math_tool.integrate_function, TOOL_INFO_ATTR)
    assert info.name == "integrate_function"
    assert info.source == "local"
    assert info.tool_path == "tools.LocalTool.math_tool.integrate_function"
    assert info.description == "积分计算工具：对 func_str 在 [a, b] 上定积分。"


def test_integrate_function_still_callable():
    result = math_tool.integrate_function("x", 0.0, 1.0)
    assert result == pytest.approx(0.5)


def test_local_tool_appears_in_isolated_tool_box_schema(math_tool_box):
    names = {schema["function"]["name"] for schema in math_tool_box.list_tools()}
    assert names == {"integrate_function"}


@pytest.mark.asyncio
async def test_local_tool_invoked_via_isolated_tool_box(math_tool_box):
    result = await math_tool_box.ainvoke(
        "integrate_function",
        {"func_str": "x**2", "a": 0.0, "b": 1.0},
    )
    assert result.error is None
    assert result.output == pytest.approx(1 / 3)
    assert result.source == "local"
