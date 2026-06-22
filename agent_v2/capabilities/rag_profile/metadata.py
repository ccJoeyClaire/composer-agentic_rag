"""Metadata written by the rag_profile capability."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from agent_v2.capabilities.rag_profile.profile import RagProfile

PROFILE_SELECTED_KEY = "profile_selected"
PROFILE_REASON_KEY = "profile_reason"
PROFILE_VALIDATED_KEY = "profile_validated"


class RagProfileMeta(TypedDict, total=False):
    """Profile the LLM chose and the router validated."""

    profile_selected: RagProfile
    profile_reason: str | None
    profile_validated: bool
