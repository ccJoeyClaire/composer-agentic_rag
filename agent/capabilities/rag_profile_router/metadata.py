"""Metadata written by the rag_profile_router capability."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from agent.capabilities.rag_profile_router.profile import RagProfile

PROFILE_SELECTED_KEY = "profile_selected"
PROFILE_REASON_KEY = "profile_reason"
PROFILE_VALIDATED_KEY = "profile_validated"


class RagProfileRouterMeta(TypedDict, total=False):
    """Profile the LLM chose and the router validated."""

    profile_selected: RagProfile
    profile_reason: str | None
    profile_validated: bool
