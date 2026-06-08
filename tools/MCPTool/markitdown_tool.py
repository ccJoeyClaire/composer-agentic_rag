from __future__ import annotations

import os
from typing import Annotated

from pydantic import Field

from tools.MCPTool._client import call_mcp_tool, mcp_command_available
from tools.MCPTool._config import markitdown_config, markitdown_ocr_config
from tools.registry import mcp_tool


@mcp_tool
async def convert_document(
    uri: Annotated[
        str,
        Field(description="待转换资源 URI，支持 http/https/file/data"),
    ],
) -> str:
    """将文档转换为 Markdown（Markitdown MCP，标准模式）。"""
    config = markitdown_config()
    if not mcp_command_available(config):
        return (
            f"未找到 Markitdown MCP 命令 `{config.command}`。"
            "请运行: pip install markitdown-mcp"
        )
    return await call_mcp_tool(
        config,
        "convert_to_markdown",
        {"uri": uri},
    )


@mcp_tool
async def convert_with_ocr(
    uri: Annotated[
        str,
        Field(description="待转换资源 URI，支持 http/https/file/data"),
    ],
) -> str:
    """将文档转换为 Markdown（Markitdown MCP + OCR 插件，适合扫描件与嵌入图片）。"""
    config = markitdown_ocr_config()
    if not mcp_command_available(config):
        return (
            f"未找到 Markitdown MCP 命令 `{config.command}`。"
            "请运行: pip install markitdown-mcp markitdown-ocr"
        )
    llm_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not llm_key:
        return (
            "OCR 模式需要 LLM Vision。"
            "请设置 OPENAI_API_KEY 或 LLM_API_KEY，并安装 markitdown-ocr。"
        )
    return await call_mcp_tool(
        config,
        "convert_to_markdown",
        {"uri": uri},
    )
