"""Tests for agent_v2 rag_profile_router validation."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent_v2.capabilities.rag_profile_router.config import RagProfileRouterConfig
from agent_v2.capabilities.rag_profile_router.node import (
    _extract_profile_from_args,
    _validate_profile,
    rag_profile_router_node,
)
from agent_v2.capabilities.rag_profile_router.profile import (
    PROFILE_USE_CONTEXTUAL_KEY,
    PROFILE_USE_HYDE_KEY,
    PROFILE_USE_RERANKER_KEY,
    PROFILE_USE_SMALL_TO_BIG_KEY,
    RagProfile,
)
from agent_v2.config import AgentConfig
from agent_v2.core.constants import DEFAULT_RAG_TOOL_NAME
from agent_v2.core.metadata.base import RAG_PROFILE_KEY
from agent_v2.core.state import AgentState

pytestmark = pytest.mark.unit


@pytest.fixture
def llm_stub():
    from unittest.mock import MagicMock

    return MagicMock()


class TestExtractProfileFromArgs:
    def test_extracts_all_search_keys(self) -> None:
        args = {
            "query": "q",
            PROFILE_USE_CONTEXTUAL_KEY: True,
            PROFILE_USE_SMALL_TO_BIG_KEY: False,
            PROFILE_USE_HYDE_KEY: True,
            PROFILE_USE_RERANKER_KEY: False,
            "recall_n": 20,
            "top_k": 5,
        }
        profile = _extract_profile_from_args(args)
        assert profile[PROFILE_USE_CONTEXTUAL_KEY] is True
        assert profile[PROFILE_USE_HYDE_KEY] is True
        assert profile["recall_n"] == 20
        assert profile["top_k"] == 5

    def test_skips_null_values(self) -> None:
        profile = _extract_profile_from_args(
            {"query": "q", PROFILE_USE_HYDE_KEY: None, PROFILE_USE_RERANKER_KEY: True}
        )
        assert PROFILE_USE_HYDE_KEY not in profile
        assert profile[PROFILE_USE_RERANKER_KEY] is True


class TestValidateProfile:
    def test_merges_baseline_defaults_when_args_empty(self) -> None:
        config = RagProfileRouterConfig(profile_id="baseline")
        validated = _validate_profile({}, config)
        assert validated[PROFILE_USE_CONTEXTUAL_KEY] is True
        assert validated[PROFILE_USE_RERANKER_KEY] is True
        assert validated[PROFILE_USE_HYDE_KEY] is False

    def test_clamps_recall_n(self) -> None:
        config = RagProfileRouterConfig(max_recall_n=10)
        validated = _validate_profile({"recall_n": 999}, config)
        assert validated["recall_n"] == 10

    def test_disables_gated_bool(self) -> None:
        config = RagProfileRouterConfig(allow_hyde=False)
        validated = _validate_profile({PROFILE_USE_HYDE_KEY: True}, config)
        assert validated[PROFILE_USE_HYDE_KEY] is False


@pytest.mark.asyncio
async def test_rag_profile_router_writes_metadata(llm_stub) -> None:
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DEFAULT_RAG_TOOL_NAME,
                        "args": {
                            "query": "agentic rag",
                            PROFILE_USE_HYDE_KEY: True,
                            PROFILE_USE_RERANKER_KEY: True,
                        },
                        "id": "c1",
                    }
                ],
            )
        ],
        "metadata": {},
    }
    agent_config = AgentConfig(llm=llm_stub, enable_rag_profile_router=True)
    capability_config = RagProfileRouterConfig(profile_id="baseline")

    update = await rag_profile_router_node(
        state,
        agent_config=agent_config,
        capability_config=capability_config,
    )

    meta = update.get("metadata") or {}
    profile: RagProfile = meta[RAG_PROFILE_KEY]
    assert meta["profile_validated"] is True
    assert profile[PROFILE_USE_HYDE_KEY] is True
    assert profile[PROFILE_USE_RERANKER_KEY] is True
    assert profile[PROFILE_USE_CONTEXTUAL_KEY] is True
