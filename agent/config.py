"""Agent configuration and capability toggles."""

from __future__ import annotations

from dataclasses import dataclass

from agent.capabilities.human_feedback.config import HumanFeedbackConfig
from agent.capabilities.rag_profile_router.config import RagProfileRouterConfig
from agent.capabilities.retrieval_gate.config import RetrievalGateConfig
from agent.core.tool_box import DEFAULT_RAG_TOOL_NAME, DEFAULT_WEB_TOOL_NAME
from agent.prompt.load import default_system_prompt_key
from llm.client import LLMClient
from tools.tool_box import ToolBox


@dataclass
class AgentConfig:
    """Runtime configuration for building an agent graph.

    Capabilities are independent toggles — not mutually exclusive patterns.
    The LLM remains the decision hub; capabilities only add nodes/tools when
    enabled.
    """

    llm: LLMClient
    tool_box: ToolBox | None = None
    checkpointer: object | None = None

    # Capability toggles
    enable_rag_profile_router: bool = False
    enable_retrieval_gate: bool = False
    enable_human_feedback: bool = False
    enable_web_search: bool = True

    # Shared tool names
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME
    web_tool_name: str = DEFAULT_WEB_TOOL_NAME
    rag_context_max_chunks: int | None = None

    # Per-capability config (optional overrides)
    rag_profile_router: RagProfileRouterConfig | None = None
    retrieval_gate: RetrievalGateConfig | None = None
    human_feedback: HumanFeedbackConfig | None = None

    # System prompt (key into ``agent/prompt/system_prompt.yaml``)
    system_prompt_key: str = default_system_prompt_key()


def eval_config(llm: LLMClient, *, tool_box: ToolBox | None = None) -> AgentConfig:
    """Preset for offline evaluation — human feedback off, query-only RAG."""
    return AgentConfig(
        llm=llm,
        tool_box=tool_box,
        enable_human_feedback=False,
        enable_rag_profile_router=False,
    )
