"""Canonical RAG profile shapes — aligned with ``arg_config.yaml`` profiles + retriever nums.

Run (from repo root):
  python -c "from rag.profile_schema import default_search_profile; print(default_search_profile())"
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from rag.config import (
    DEFAULT_INDEX_PROFILE_ID,
    DEFAULT_RETRIEVE_PROFILE_ID,
    IndexProfileConfig,
    RetrieveProfileConfig,
    get_index_profile,
    get_rag_config,
    get_retrieve_profile,
)

USE_TOKEN_CHUNKER_KEY = "use_token_chunker"
USE_CONTEXTUAL_KEY = "use_contextual"
USE_SMALL_TO_BIG_KEY = "use_small_to_big"
USE_PREDICT_QUESTIONS_KEY = "use_predict_questions"
USE_HYDE_KEY = "use_hyde"
USE_RERANKER_KEY = "use_reranker"
RECALL_N_KEY = "recall_n"
TOP_K_KEY = "top_k"

INDEX_BOOL_KEYS = (
    USE_TOKEN_CHUNKER_KEY,
    USE_CONTEXTUAL_KEY,
    USE_SMALL_TO_BIG_KEY,
    USE_PREDICT_QUESTIONS_KEY,
)

SEARCH_BOOL_KEYS = (
    USE_CONTEXTUAL_KEY,
    USE_SMALL_TO_BIG_KEY,
    USE_HYDE_KEY,
    USE_RERANKER_KEY,
)

SEARCH_NUM_KEYS = (RECALL_N_KEY, TOP_K_KEY)

SEARCH_PROFILE_KEYS = SEARCH_BOOL_KEYS + SEARCH_NUM_KEYS
INDEX_PROFILE_KEYS = INDEX_BOOL_KEYS


class RagIndexProfile(TypedDict, total=False):
    """Index-time pipeline options (``build_RAG_indexer``)."""

    use_token_chunker: bool
    use_contextual: bool
    use_small_to_big: bool
    use_predict_questions: bool


class RagSearchProfile(TypedDict, total=False):
    """Query-time pipeline options (``build_RAG_retriever`` + ``top_k``)."""

    use_contextual: bool
    use_small_to_big: bool
    use_hyde: bool
    use_reranker: bool
    recall_n: int
    top_k: int


def _search_bools_from_retrieve_profile(
    profile: RetrieveProfileConfig,
) -> RagSearchProfile:
    return {
        USE_CONTEXTUAL_KEY: profile.use_contextual,
        USE_SMALL_TO_BIG_KEY: profile.use_small_to_big,
        USE_HYDE_KEY: profile.use_hyde,
        USE_RERANKER_KEY: profile.use_reranker,
    }


def _index_bools_from_index_profile(profile: IndexProfileConfig) -> RagIndexProfile:
    return {
        USE_TOKEN_CHUNKER_KEY: profile.use_token_chunker,
        USE_CONTEXTUAL_KEY: profile.use_contextual,
        USE_SMALL_TO_BIG_KEY: profile.use_small_to_big,
        USE_PREDICT_QUESTIONS_KEY: profile.use_predict_questions,
    }


def default_search_profile(
    retrieve_profile_id: str = DEFAULT_RETRIEVE_PROFILE_ID,
) -> RagSearchProfile:
    """Bool defaults from ``retrieve_profiles.<id>``; nums from retriever config."""
    yaml_profile = get_retrieve_profile(get_rag_config(), retrieve_profile_id)
    retriever_cfg = get_rag_config().retriever
    merged: RagSearchProfile = _search_bools_from_retrieve_profile(yaml_profile)
    merged[RECALL_N_KEY] = retriever_cfg.recall_n
    merged[TOP_K_KEY] = retriever_cfg.top_k
    return merged


def default_index_profile(
    index_profile_id: str = DEFAULT_INDEX_PROFILE_ID,
) -> RagIndexProfile:
    """Bool defaults from ``index_profiles.<id>``."""
    return _index_bools_from_index_profile(
        get_index_profile(get_rag_config(), index_profile_id)
    )


def merge_search_profile(
    overrides: RagSearchProfile | None,
    *,
    defaults: RagSearchProfile,
) -> RagSearchProfile:
    """Overlay tool-call args onto deployment defaults."""
    merged: RagSearchProfile = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


def merge_index_profile(
    overrides: RagIndexProfile | None,
    *,
    defaults: RagIndexProfile,
) -> RagIndexProfile:
    merged: RagIndexProfile = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


def normalize_search_profile(
    profile: RagSearchProfile | None,
    *,
    defaults: RagSearchProfile,
    max_recall_n: int,
    max_top_k: int | None = None,
) -> tuple[RagSearchProfile, list[str]]:
    """Merge defaults and clamp numeric fields."""
    notes: list[str] = []
    validated = merge_search_profile(profile, defaults=defaults)

    recall = validated.get(RECALL_N_KEY, defaults.get(RECALL_N_KEY, 1))
    if recall < 1:
        recall = 1
    if recall > max_recall_n:
        notes.append(f"recall_n {recall} exceeds max {max_recall_n}; clamped.")
        recall = max_recall_n
    validated[RECALL_N_KEY] = recall

    top_k = validated.get(TOP_K_KEY, defaults.get(TOP_K_KEY, 1))
    if top_k < 1:
        top_k = 1
    if max_top_k is not None and top_k > max_top_k:
        notes.append(f"top_k {top_k} exceeds max {max_top_k}; clamped.")
        top_k = max_top_k
    validated[TOP_K_KEY] = top_k

    return validated, notes


def normalize_index_profile(
    profile: RagIndexProfile | None,
    *,
    defaults: RagIndexProfile,
) -> tuple[RagIndexProfile, list[str]]:
    """Merge index defaults with tool-call overrides."""
    return merge_index_profile(profile, defaults=defaults), []


def search_profile_from_optional_args(
    *,
    use_contextual: bool | None = None,
    use_small_to_big: bool | None = None,
    use_hyde: bool | None = None,
    use_reranker: bool | None = None,
    recall_n: int | None = None,
    top_k: int | None = None,
) -> RagSearchProfile | None:
    """Build a partial search profile from tool args (``None`` = omit / use defaults)."""
    profile: RagSearchProfile = {}
    if use_contextual is not None:
        profile[USE_CONTEXTUAL_KEY] = use_contextual
    if use_small_to_big is not None:
        profile[USE_SMALL_TO_BIG_KEY] = use_small_to_big
    if use_hyde is not None:
        profile[USE_HYDE_KEY] = use_hyde
    if use_reranker is not None:
        profile[USE_RERANKER_KEY] = use_reranker
    if recall_n is not None:
        profile[RECALL_N_KEY] = recall_n
    if top_k is not None:
        profile[TOP_K_KEY] = top_k
    return profile or None


def index_profile_from_optional_args(
    *,
    use_token_chunker: bool | None = None,
    use_contextual: bool | None = None,
    use_small_to_big: bool | None = None,
    use_predict_questions: bool | None = None,
) -> RagIndexProfile | None:
    profile: RagIndexProfile = {}
    if use_token_chunker is not None:
        profile[USE_TOKEN_CHUNKER_KEY] = use_token_chunker
    if use_contextual is not None:
        profile[USE_CONTEXTUAL_KEY] = use_contextual
    if use_small_to_big is not None:
        profile[USE_SMALL_TO_BIG_KEY] = use_small_to_big
    if use_predict_questions is not None:
        profile[USE_PREDICT_QUESTIONS_KEY] = use_predict_questions
    return profile or None
