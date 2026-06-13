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
    assert len(mcp_tools) == 4
    assert "tools.MCPTool.markitdown_tool.convert_document" in paths
    assert "tools.MCPTool.markitdown_tool.convert_with_ocr" in paths
    assert "tools.MCPTool.tavily_tool.tavily_search" in paths
    assert "tools.MCPTool.tavily_tool.tavily_extract" in paths


def test_mcp_tools_schema():
    box = ToolBox()
    schemas = box.list_tools()
    names = {s["function"]["name"] for s in schemas}
    assert {"convert_document", "convert_with_ocr", "tavily_search", "tavily_extract"}.issubset(names)


@pytest.mark.asyncio
async def test_tavily_search_without_mcp_url(monkeypatch):
    monkeypatch.delenv("TAVILY_MCP_URL", raising=False)
    from tools.MCPTool.tavily_tool import tavily_search

    result = await tavily_search("测试")
    assert "TAVILY_MCP_URL" in result


@pytest.mark.asyncio
async def test_tavily_search_without_api_key(monkeypatch):
    monkeypatch.setenv(
        "TAVILY_MCP_URL",
        "https://mcp.tavily.com/mcp/?tavilyApiKey=<your-api-key>",
    )
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from tools.MCPTool.tavily_tool import tavily_search

    result = await tavily_search("测试")
    assert "TAVILY_API_KEY" in result


@pytest.mark.asyncio
async def test_tavily_extract_without_mcp_url(monkeypatch):
    monkeypatch.delenv("TAVILY_MCP_URL", raising=False)
    from tools.MCPTool.tavily_tool import tavily_extract

    result = await tavily_extract(["https://example.com"])
    assert "TAVILY_MCP_URL" in result


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


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/tools/test_mcp_tools.py -v
# ================================================================================================================
