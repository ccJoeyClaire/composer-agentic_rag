from __future__ import annotations

import os
import shutil
from typing import Any

from tools.MCPTool._config import MCPServerConfig


def _extract_tool_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    if parts:
        return "\n".join(parts)
    if getattr(result, "isError", False):
        return str(result)
    return ""


def mcp_command_available(config: MCPServerConfig) -> bool:
    return shutil.which(config.command) is not None


async def call_mcp_tool(
    config: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        return "MCP SDK 未安装。请运行: pip install mcp>=1.6.0"

    if not mcp_command_available(config):
        return (
            f"未找到 MCP 命令 `{config.command}`。"
            f"请安装对应 server 或设置环境变量覆盖启动命令。"
        )

    env = {**os.environ, **config.env}
    server_params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=env,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                text = _extract_tool_text(result)
                if text:
                    return text
                if getattr(result, "isError", False):
                    return f"MCP 工具 {tool_name} 调用失败"
                return ""
    except Exception as e:
        return f"MCP 调用失败 ({config.key}/{tool_name}): {e}"
