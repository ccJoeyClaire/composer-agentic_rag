"""Build a compiled agent graph from :class:`RequestConfig`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from agent.builder import build_agent
from agent.capabilities.rag_profile_router.config import RagProfileRouterConfig
from agent.config import AgentConfig
from agent.pattern.config import PatternConfig, get_pattern
from llm.client import LLMClient
from rag.context import bind_rag_context
from tools.tool_box import ToolBox


@dataclass(frozen=True)
class RequestConfig:
    """Deployment + pattern selection for one agent run (no user query)."""

    pattern_id: str
    collection: str
    profile_id: str
    enable_web_search: bool = False
    config_path: Path | None = None


def _agent_config_from_request(
    request: RequestConfig,
    pattern: PatternConfig,
    *,
    llm: LLMClient,
    tool_box: ToolBox,
) -> AgentConfig:
    return AgentConfig(
        llm=llm,
        tool_box=tool_box,
        enable_rag_profile_router=pattern.enable_rag_profile_router,
        enable_retrieval_gate=pattern.enable_retrieval_gate,
        enable_human_feedback=pattern.enable_human_feedback,
        enable_web_search=request.enable_web_search,
        rag_profile_router=RagProfileRouterConfig(profile_id=request.profile_id),
    )


def build_graph(request: RequestConfig) -> CompiledStateGraph:
    """Bind RAG, assemble ``AgentConfig``, and return a compiled LangGraph."""
    pattern = get_pattern(request.pattern_id, config_path=request.config_path)
    bind_rag_context(
        collection=request.collection,
        profile_id=request.profile_id,
    )
    llm = LLMClient()
    tool_box = ToolBox()
    agent_config = _agent_config_from_request(
        request,
        pattern,
        llm=llm,
        tool_box=tool_box,
    )
    return build_agent(agent_config)
