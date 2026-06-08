from __future__ import annotations

import os
from typing import Annotated

from pydantic import Field

from tools.MCPTool._client import call_mcp_tool, mcp_command_available
from tools.MCPTool._config import bocha_config
from tools.registry import mcp_tool


@mcp_tool
async def bocha(
    query: Annotated[str, Field(description="搜索关键词")],
    freshness: Annotated[
        str,
        Field(
            description=(
                "时间范围: noLimit, oneDay, oneWeek, oneMonth, oneYear, "
                "YYYY-MM-DD 或 YYYY-MM-DD..YYYY-MM-DD"
            )
        ),
    ] = "noLimit",
    count: Annotated[int, Field(description="返回条数，1-50", ge=1, le=50)] = 10,
) -> str:
    """博查联网搜索，返回网页标题、摘要、链接与时间。"""
    if not os.environ.get("BOCHA_API_KEY"):
        return "Error: 未配置 BOCHA_API_KEY 环境变量。"

    config = bocha_config()
    if not mcp_command_available(config):
        return (
            f"未找到 Bocha MCP 命令 `{config.command}`。"
            "请运行: pip install bocha-search-mcp"
        )

    return await call_mcp_tool(
        config,
        "bocha_web_search",
        {"query": query, "freshness": freshness, "count": count},
    )
