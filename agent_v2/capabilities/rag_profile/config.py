"""Rag profile capability configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_v2.capabilities.rag_profile.profile import DEFAULT_RAG_PROFILE, RagProfile


@dataclass
class RagProfileConfig:
    """Settings for validating LLM-specified RAG profiles."""

    default_profile: RagProfile = field(default_factory=lambda: dict(DEFAULT_RAG_PROFILE))
    allow_hyde: bool = True
    allow_reranker: bool = True
    max_recall_n: int | None = None
