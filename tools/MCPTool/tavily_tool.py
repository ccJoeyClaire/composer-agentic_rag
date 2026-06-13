from __future__ import annotations

import os
from typing import Annotated, Literal

from pydantic import Field

from tools.MCPTool._client import call_mcp_tool, mcp_command_available
from tools.MCPTool._config import TAVILY_API_KEY_PLACEHOLDER, tavily_config
from tools.registry import mcp_tool

TAVILY_SEARCH_TOOL = "tavily_search"
TAVILY_EXTRACT_TOOL = "tavily_extract"


def _tavily_preflight_error() -> str | None:
    url_template = os.environ.get("TAVILY_MCP_URL", "").strip()
    if not url_template:
        return "Error: 未配置 TAVILY_MCP_URL 环境变量。"

    if (
        TAVILY_API_KEY_PLACEHOLDER in url_template
        and not os.environ.get("TAVILY_API_KEY", "").strip()
    ):
        return "Error: 未配置 TAVILY_API_KEY 环境变量。"

    config = tavily_config()
    if not config.args or not config.args[-1]:
        return "Error: 未配置 TAVILY_API_KEY 环境变量。"
    if not mcp_command_available(config):
        return (
            f"未找到 MCP 命令 `{config.command}`。"
            "请安装 Node.js（含 npx）或设置 TAVILY_MCP_COMMAND 环境变量。"
        )
    return None


@mcp_tool
async def tavily_search(
    query: Annotated[str, Field(description="搜索关键词")],
    max_results: Annotated[
        int,
        Field(description="最多返回条数", ge=1, le=20),
    ] = 5,
    search_depth: Annotated[
        Literal["basic", "advanced", "fast", "ultra-fast"],
        Field(
            description=(
                "搜索深度: basic 通用, advanced 更深入, "
                "fast 低延迟高相关, ultra-fast 优先速度"
            )
        ),
    ] = "basic",
    time_range: Annotated[
        Literal["day", "week", "month", "year"] | None,
        Field(description="时间范围: day/week/month/year；不限制则留空"),
    ] = None,
    include_domains: Annotated[
        list[str],
        Field(description="仅搜索这些域名，如 ['arxiv.org', 'nature.com']"),
    ] = [],
    exclude_domains: Annotated[
        list[str],
        Field(description="排除这些域名"),
    ] = [],
) -> str:
    """Tavily 联网搜索，适合新闻、实时信息与事实查询。"""
    preflight_error = _tavily_preflight_error()
    if preflight_error is not None:
        return preflight_error

    return await call_mcp_tool(
        tavily_config(),
        TAVILY_SEARCH_TOOL,
        {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "time_range": time_range,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
        },
    )


@mcp_tool
async def tavily_extract(
    urls: Annotated[
        list[str],
        Field(description="待抽取内容的 URL 列表", min_length=1),
    ],
    extract_depth: Annotated[
        Literal["basic", "advanced"],
        Field(
            description=(
                "抽取深度: basic 常规页面; advanced 适合 LinkedIn、"
                "受保护站点或含表格/嵌入内容的页面"
            )
        ),
    ] = "basic",
    format: Annotated[
        Literal["markdown", "text"],
        Field(description="输出格式"),
    ] = "markdown",
) -> str:
    """从 URL 抽取网页正文，适合已知链接的深度阅读与 CRAG 兜底。"""
    preflight_error = _tavily_preflight_error()
    if preflight_error is not None:
        return preflight_error

    return await call_mcp_tool(
        tavily_config(),
        TAVILY_EXTRACT_TOOL,
        {
            "urls": urls,
            "extract_depth": extract_depth,
            "format": format,
        },
    )
