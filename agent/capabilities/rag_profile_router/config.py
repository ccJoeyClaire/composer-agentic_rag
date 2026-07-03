"""Rag profile router capability configuration."""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import DEFAULT_RETRIEVE_PROFILE_ID, get_rag_config
from rag.profile_schema import RagSearchProfile, default_search_profile

from agent.capabilities.rag_profile_router.profile import RagProfile


@dataclass
class RagProfileRouterConfig:
    """Settings for validating LLM-specified RAG search profile overrides."""

    retrieve_profile_id: str = DEFAULT_RETRIEVE_PROFILE_ID
    default_profile: RagProfile | None = None
    max_recall_n: int | None = None
    max_top_k: int | None = None

    def resolved_defaults(self) -> RagSearchProfile:
        """Deployment default profile (explicit override or yaml retrieve profile)."""
        if self.default_profile is not None:
            return dict(self.default_profile)
        return default_search_profile(self.retrieve_profile_id)

    def resolved_max_recall_n(self) -> int:
        if self.max_recall_n is not None:
            return self.max_recall_n
        return get_rag_config().retriever.recall_n
