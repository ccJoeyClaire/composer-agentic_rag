"""RAG profile shape — LLM specifies values in tool-call args; router validates them."""

from __future__ import annotations

from typing import NotRequired, TypedDict

PROFILE_USE_HYDE_KEY = "use_hyde"
PROFILE_USE_RERANKER_KEY = "use_reranker"
PROFILE_RECALL_N_KEY = "recall_n"
PROFILE_TOP_K_KEY = "top_k"


class RagProfile(TypedDict, total=False):
    """Query-time retrieval options the LLM may attach to a RAG tool call."""

    use_hyde: bool
    use_reranker: bool
    recall_n: int
    top_k: int


DEFAULT_RAG_PROFILE: RagProfile = {
    PROFILE_USE_HYDE_KEY: False,
    PROFILE_USE_RERANKER_KEY: True,
}
