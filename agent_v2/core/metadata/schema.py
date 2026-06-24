"""Composed agent metadata — the only module that imports all capability meta schemas."""

from __future__ import annotations

from typing import TypedDict

from agent_v2.capabilities.human_feedback.metadata import HumanFeedbackMeta
from agent_v2.capabilities.rag_profile_router.metadata import RagProfileRouterMeta
from agent_v2.capabilities.retrieval_gate.metadata import RetrievalGateMeta
from agent_v2.core.metadata.base import RagInvocationMeta


class AgentMetadata(
    RagInvocationMeta,
    RagProfileRouterMeta,
    RetrievalGateMeta,
    HumanFeedbackMeta,
    total=False,
):
    """All metadata keys that may appear on ``AgentState.metadata``."""
