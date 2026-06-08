"""Unit tests for MCP tool wrappers."""

from __future__ import annotations

import pytest

from tools.registry import discover_packages, get_decorated_tools
from tools.tool_box import ToolBox

pytestmark = pytest.mark.unit


def test_mcp_registry_paths():
    discover_packages("tools.MCPTool")
    mcp_tools = [t for t in get_decorated_tools().values() if t.source == "mcp"]
    paths = {t.tool_path for t in mcp_tools}
    assert len(mcp_tools) == 3
    assert "tools.MCPTool.markitdown_tool.convert_document" in paths
    assert "tools.MCPTool.markitdown_tool.convert_with_ocr" in paths
    assert "tools.MCPTool.search_tool.bocha" in paths


def test_mcp_tools_schema():
    box = ToolBox()
    schemas = box.list_tools()
    names = {s["function"]["name"] for s in schemas}
    assert {"convert_document", "convert_with_ocr", "bocha"}.issubset(names)


@pytest.mark.asyncio
async def test_bocha_without_api_key(monkeypatch):
    monkeypatch.delenv("BOCHA_API_KEY", raising=False)
    from tools.MCPTool.search_tool import bocha

    result = await bocha("测试")
    assert "BOCHA_API_KEY" in result


@pytest.mark.asyncio
async def test_convert_document_without_markitdown(monkeypatch):
    monkeypatch.setenv("MARKITDOWN_MCP_COMMAND", "__missing_markitdown_mcp__")
    from tools.MCPTool.markitdown_tool import convert_document

    result = await convert_document("file:///tmp/test.pdf")
    assert "未找到 Markitdown MCP 命令" in result


@pytest.mark.asyncio
async def test_convert_with_ocr_without_llm_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        "tools.MCPTool.markitdown_tool.mcp_command_available",
        lambda _config: True,
    )
    from tools.MCPTool.markitdown_tool import convert_with_ocr

    result = await convert_with_ocr("file:///tmp/scan.pdf")
    assert "OCR" in result or "OPENAI_API_KEY" in result or "LLM_API_KEY" in result
