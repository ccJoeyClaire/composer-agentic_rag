"""Unit tests for ToolContextBundle and ToolBox context injection."""

from __future__ import annotations

import pytest

from tools.context import INJECTED_CONTEXT_PARAM, ToolContextBundle, ToolContextView
from tools.registry import TOOL_INFO_ATTR, local_tool
from tools.tool_box import ToolBox

pytestmark = pytest.mark.unit

DEMO_CONTEXT_KEY = "demo"


@local_tool(context_keys=(DEMO_CONTEXT_KEY,))
def _echo_with_context(
    message: str,
    *,
    _tool_context: ToolContextView | None = None,
) -> str:
    """Echo message with injected deployment prefix."""
    assert _tool_context is not None
    return _tool_context.require(DEMO_CONTEXT_KEY, str) + ":" + message


def test_tool_context_bundle_bind_and_require():
    bundle = ToolContextBundle()
    bundle.bind(DEMO_CONTEXT_KEY, "deploy")
    assert bundle.require(DEMO_CONTEXT_KEY, str) == "deploy"


@pytest.mark.asyncio
async def test_tool_box_injects_context():
    bundle = ToolContextBundle()
    bundle.bind(DEMO_CONTEXT_KEY, "deploy")
    box = ToolBox(autodiscover=False, context=bundle)
    info = getattr(_echo_with_context, TOOL_INFO_ATTR)
    box._registry[info.name] = info

    result = await box.ainvoke(info.name, {"message": "hi"})
    assert result.error is None
    assert result.output == "deploy:hi"


@pytest.mark.asyncio
async def test_tool_box_missing_context_returns_error():
    box = ToolBox(autodiscover=False)
    info = getattr(_echo_with_context, TOOL_INFO_ATTR)
    box._registry[info.name] = info

    result = await box.ainvoke(info.name, {"message": "hi"})
    assert result.error == f"Missing tool context: {DEMO_CONTEXT_KEY}"


def test_list_tools_strips_injected_param():
    bundle = ToolContextBundle()
    bundle.bind(DEMO_CONTEXT_KEY, "deploy")
    box = ToolBox(autodiscover=False, context=bundle)
    info = getattr(_echo_with_context, TOOL_INFO_ATTR)
    box._registry[info.name] = info

    schema = box.list_tools()[0]
    props = schema["function"]["parameters"]["properties"]
    assert INJECTED_CONTEXT_PARAM not in props
    assert "message" in props
