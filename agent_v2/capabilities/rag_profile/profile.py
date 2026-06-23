"""RAG search profile — canonical shape lives in :mod:`rag.profile_schema`."""

from __future__ import annotations

from rag.config import DEFAULT_PROFILE_ID
from rag.profile_schema import (
    RECALL_N_KEY,
    SEARCH_PROFILE_KEYS,
    TOP_K_KEY,
    USE_CONTEXTUAL_KEY,
    USE_HYDE_KEY,
    USE_RERANKER_KEY,
    USE_SMALL_TO_BIG_KEY,
    RagSearchProfile,
    default_search_profile,
)

RagProfile = RagSearchProfile

DEFAULT_RAG_PROFILE: RagProfile = default_search_profile(DEFAULT_PROFILE_ID)

PROFILE_USE_CONTEXTUAL_KEY = USE_CONTEXTUAL_KEY
PROFILE_USE_SMALL_TO_BIG_KEY = USE_SMALL_TO_BIG_KEY
PROFILE_USE_HYDE_KEY = USE_HYDE_KEY
PROFILE_USE_RERANKER_KEY = USE_RERANKER_KEY
PROFILE_RECALL_N_KEY = RECALL_N_KEY
PROFILE_TOP_K_KEY = TOP_K_KEY

__all__ = [
    "DEFAULT_PROFILE_ID",
    "DEFAULT_RAG_PROFILE",
    "PROFILE_RECALL_N_KEY",
    "PROFILE_TOP_K_KEY",
    "PROFILE_USE_CONTEXTUAL_KEY",
    "PROFILE_USE_HYDE_KEY",
    "PROFILE_USE_RERANKER_KEY",
    "PROFILE_USE_SMALL_TO_BIG_KEY",
    "RagProfile",
    "SEARCH_PROFILE_KEYS",
]
