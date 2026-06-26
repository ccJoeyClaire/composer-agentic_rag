"""Apply LLM-specified RAG profile overrides from tool-call args."""

from __future__ import annotations

from typing import Any

from rag.profile_schema import (
    RagSearchProfile,
    SEARCH_PROFILE_KEYS,
    normalize_search_profile,
)

from agent.capabilities.rag_profile_router.config import RagProfileRouterConfig
from agent.capabilities.rag_profile_router.metadata import (
    PROFILE_REASON_KEY,
    PROFILE_SELECTED_KEY,
    PROFILE_VALIDATED_KEY,
)
from agent.capabilities.rag_profile_router.profile import RagProfile
from agent.config import AgentConfig
from agent.core.edges.tool_calls import (
    has_rag_tool_call,
    last_ai_message,
    tool_calls_from_ai,
)
from agent.core.metadata.base import (
    RAG_LAST_QUERY_KEY,
    RAG_PROFILE_KEY,
    RAG_TOOL_NAME_KEY,
)
from agent.core.state import AgentState, merge_metadata


def _extract_profile_from_args(args: dict[str, Any]) -> RagSearchProfile:
    """Pull search-profile keys present in the tool call (skip omitted / null)."""
    profile: RagSearchProfile = {}
    for key in SEARCH_PROFILE_KEYS:
        if key in args and args[key] is not None:
            profile[key] = args[key]  # type: ignore[literal-required]
    return profile


def _validate_profile(
    profile: RagSearchProfile,
    config: RagProfileRouterConfig,
) -> RagProfile:
    """Merge defaults, apply deployment gates, and clamp numeric fields."""
    validated, _ = normalize_search_profile(
        profile or None,
        defaults=config.resolved_defaults(),
        allow_contextual=config.allow_contextual,
        allow_small_to_big=config.allow_small_to_big,
        allow_hyde=config.allow_hyde,
        allow_reranker=config.allow_reranker,
        max_recall_n=config.resolved_max_recall_n(),
        max_top_k=config.max_top_k,
    )
    return validated


async def rag_profile_router_node(
    state: AgentState,
    *,
    agent_config: AgentConfig,
    capability_config: RagProfileRouterConfig,
) -> dict[str, object]:
    """Read RAG profile overrides + query from the LLM's tool call; write metadata.

    Only runs when ``enable_rag_profile_router`` is on — the LLM may override
    pipeline flags beyond ``query`` in that mode.
    """
    ai_msg = last_ai_message(state["messages"])
    if ai_msg is None:
        return {}

    tool_calls = tool_calls_from_ai(ai_msg)
    if not has_rag_tool_call(tool_calls, rag_tool_name=agent_config.rag_tool_name):
        return {}

    rag_call = next(
        c for c in tool_calls if c.get("name") == agent_config.rag_tool_name
    )
    args = rag_call.get("args") or {}
    raw_profile = _extract_profile_from_args(args)
    profile = _validate_profile(raw_profile, capability_config)

    return merge_metadata(
        state,
        {
            RAG_TOOL_NAME_KEY: agent_config.rag_tool_name,
            RAG_LAST_QUERY_KEY: str(args.get("query", "")),
            RAG_PROFILE_KEY: profile,
            PROFILE_SELECTED_KEY: profile,
            PROFILE_VALIDATED_KEY: True,
            PROFILE_REASON_KEY: "llm_tool_call",
        },
    )
