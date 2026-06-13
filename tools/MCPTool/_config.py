from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    key: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def _command_from_env(env_var: str, default: str) -> tuple[str, list[str]]:
    raw = os.environ.get(env_var, default).strip()
    parts = shlex.split(raw)
    if not parts:
        raise ValueError(f"{env_var} 不能为空")
    return parts[0], parts[1:]


def markitdown_config() -> MCPServerConfig:
    command, args = _command_from_env("MARKITDOWN_MCP_COMMAND", "markitdown-mcp")
    return MCPServerConfig(key="markitdown", command=command, args=args)


def markitdown_ocr_config() -> MCPServerConfig:
    command, args = _command_from_env("MARKITDOWN_MCP_COMMAND", "markitdown-mcp")
    env = {"MARKITDOWN_ENABLE_PLUGINS": "true"}
    llm_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if llm_key:
        env["OPENAI_API_KEY"] = llm_key
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    model = os.environ.get("MARKITDOWN_OCR_MODEL") or os.environ.get("LLM_MODEL_ID")
    if model:
        env["MARKITDOWN_LLM_MODEL"] = model
    return MCPServerConfig(key="markitdown_ocr", command=command, args=args, env=env)


def bocha_config() -> MCPServerConfig:
    command, args = _command_from_env("BOCHA_MCP_COMMAND", "bocha-search-mcp")
    env: dict[str, str] = {}
    api_key = os.environ.get("BOCHA_API_KEY", "")
    if api_key:
        env["BOCHA_API_KEY"] = api_key
    return MCPServerConfig(key="bocha", command=command, args=args, env=env)


TAVILY_API_KEY_PLACEHOLDER = "<your-api-key>"


def tavily_config() -> MCPServerConfig:
    """Stdio bridge to Tavily remote MCP via ``npx mcp-remote``."""
    command, args = _command_from_env("TAVILY_MCP_COMMAND", "npx -y mcp-remote")
    url_template = os.environ.get("TAVILY_MCP_URL", "").strip()
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if TAVILY_API_KEY_PLACEHOLDER in url_template:
        remote_url = url_template.replace(TAVILY_API_KEY_PLACEHOLDER, api_key)
    else:
        remote_url = url_template

    return MCPServerConfig(
        key="tavily",
        command=command,
        args=[*args, remote_url],
    )
