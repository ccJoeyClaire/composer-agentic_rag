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


def _main() -> None:
    """Markitdown MCP tool smoke test.

    Run (from repo root):
      python -m tools.MCPTool.markitdown_tool convert file:///path/to/doc.pdf
      python -m tools.MCPTool.markitdown_tool ocr file:///path/to/scan.pdf
    """
    import argparse
    import asyncio
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Markitdown MCP tool CLI.")
    sub = parser.add_subparsers(dest="command", required=True)
    convert_p = sub.add_parser("convert")
    convert_p.add_argument("uri")
    ocr_p = sub.add_parser("ocr")
    ocr_p.add_argument("uri")
    args = parser.parse_args()

    if args.command == "convert":
        result = asyncio.run(convert_document(args.uri))
    else:
        result = asyncio.run(convert_with_ocr(args.uri))

    if result.startswith("未找到") or result.startswith("OCR 模式"):
        print(result, file=sys.stderr)
        sys.exit(1)
    print(result)


if __name__ == "__main__":
    _main()
