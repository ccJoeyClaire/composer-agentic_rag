"""Composed agent metadata — the only module that imports all capability meta schemas."""

from __future__ import annotations

from typing import TypedDict

from agent.capabilities.human_feedback.metadata import HumanFeedbackMeta
from agent.capabilities.rag_profile_router.metadata import RagProfileRouterMeta
from agent.capabilities.retrieval_gate.metadata import RetrievalGateMeta
from agent.core.metadata.base import RagInvocationMeta


class AgentMetadata(
    RagInvocationMeta,
    RagProfileRouterMeta,
    RetrievalGateMeta,
    HumanFeedbackMeta,
    total=False,
):
    """All metadata keys that may appear on ``AgentState.metadata``."""
