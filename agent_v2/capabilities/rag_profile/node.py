"""Apply LLM-specified RAG profile from tool-call args (stub)."""

from __future__ import annotations

from typing import Any

from agent_v2.capabilities.rag_profile.config import RagProfileConfig
from agent_v2.capabilities.rag_profile.metadata import (
    PROFILE_REASON_KEY,
    PROFILE_SELECTED_KEY,
    PROFILE_VALIDATED_KEY,
)
from agent_v2.capabilities.rag_profile.profile import (
    PROFILE_RECALL_N_KEY,
    PROFILE_TOP_K_KEY,
    PROFILE_USE_HYDE_KEY,
    PROFILE_USE_RERANKER_KEY,
    RagProfile,
)
from agent_v2.core.edges.tool_calls import (
    has_rag_tool_call,
    last_ai_message,
    tool_calls_from_ai,
)
from agent_v2.core.metadata.base import (
    RAG_LAST_QUERY_KEY,
    RAG_PROFILE_KEY,
    RAG_TOOL_NAME_KEY,
)
from agent_v2.core.state import AgentState, merge_metadata
from agent_v2.config import AgentConfig


_PROFILE_ARG_KEYS = (
    PROFILE_USE_HYDE_KEY,
    PROFILE_USE_RERANKER_KEY,
    PROFILE_RECALL_N_KEY,
    PROFILE_TOP_K_KEY,
)


def _extract_profile_from_args(args: dict[str, Any]) -> RagProfile:
    profile: RagProfile = {}
    for key in _PROFILE_ARG_KEYS:
        if key in args:
            profile[key] = args[key]  # type: ignore[literal-required]
    return profile


def _validate_profile(profile: RagProfile, config: RagProfileConfig) -> RagProfile:
    """Clamp profile fields to deployment allow-range (stub — no LLM logic)."""
    validated: RagProfile = dict(config.default_profile)
    validated.update(profile)

    if not config.allow_hyde:
        validated[PROFILE_USE_HYDE_KEY] = False
    if not config.allow_reranker:
        validated[PROFILE_USE_RERANKER_KEY] = False
    if config.max_recall_n is not None and PROFILE_RECALL_N_KEY in validated:
        validated[PROFILE_RECALL_N_KEY] = min(
            validated[PROFILE_RECALL_N_KEY],
            config.max_recall_n,
        )
    return validated


async def rag_profile_router_node(
    state: AgentState,
    *,
    agent_config: AgentConfig,
    capability_config: RagProfileConfig,
) -> dict[str, object]:
    """Read RAG profile + query from the LLM's tool call; write metadata.

    Does not choose the profile — the LLM already did in ``tool_calls``.
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
