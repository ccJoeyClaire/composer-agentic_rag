"""Tests for agent_v2 pattern yaml loading."""

from __future__ import annotations

import pytest

from agent_v2.pattern.config import get_pattern

pytestmark = pytest.mark.unit


def test_crag_pattern_enables_gate_only() -> None:
    pattern = get_pattern("crag")
    assert pattern.enable_retrieval_gate is True
    assert pattern.enable_rag_profile_router is False
    assert pattern.enable_human_feedback is False


def test_self_rag_pattern_enables_router_only() -> None:
    pattern = get_pattern("self_rag")
    assert pattern.enable_retrieval_gate is False
    assert pattern.enable_rag_profile_router is True
    assert pattern.enable_human_feedback is False


def test_full_pattern_enables_all() -> None:
    pattern = get_pattern("full")
    assert pattern.enable_retrieval_gate is True
    assert pattern.enable_rag_profile_router is True
    assert pattern.enable_human_feedback is True


def test_unknown_pattern_raises() -> None:
    with pytest.raises(KeyError, match="unknown"):
        get_pattern("unknown")
