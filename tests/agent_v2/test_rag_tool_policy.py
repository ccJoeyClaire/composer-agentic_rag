"""Tests for agent_v2 RAG tool policy (query-only vs override schema)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_v2.core.constants import DEFAULT_RAG_TOOL_NAME
from agent_v2.core.rag_tool_policy import (
    RAG_SEARCH_QUERY_ARG,
    rag_search_args_query_only,
    restrict_rag_search_tool_schema,
)
from agent_v2.core.tool_box import AgentToolBox
from rag.profile_schema import USE_HYDE_KEY
from tools.tool_box import ToolBox

pytestmark = pytest.mark.unit


@pytest.fixture
def rag_search_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": DEFAULT_RAG_TOOL_NAME,
            "parameters": {
                "type": "object",
                "properties": {
                    RAG_SEARCH_QUERY_ARG: {"type": "string"},
                    USE_HYDE_KEY: {"type": "boolean"},
                    "recall_n": {"type": "integer"},
                },
                "required": [RAG_SEARCH_QUERY_ARG],
            },
        },
    }


class TestRagToolPolicy:
    def test_restrict_schema_keeps_query_only(self, rag_search_schema: dict) -> None:
        restricted = restrict_rag_search_tool_schema(
            rag_search_schema,
            rag_tool_name=DEFAULT_RAG_TOOL_NAME,
        )
        props = restricted["function"]["parameters"]["properties"]
        assert set(props) == {RAG_SEARCH_QUERY_ARG}

    def test_restrict_schema_leaves_other_tools_unchanged(self, rag_search_schema: dict) -> None:
        other = {
            "type": "function",
            "function": {
                "name": "math_tool",
                "parameters": {"properties": {"expression": {"type": "string"}}},
            },
        }
        assert restrict_rag_search_tool_schema(other, rag_tool_name=DEFAULT_RAG_TOOL_NAME) == other

    def test_query_only_args_strip_profile_keys(self) -> None:
        stripped = rag_search_args_query_only(
            {"query": "q", USE_HYDE_KEY: True, "recall_n": 99}
        )
        assert stripped == {"query": "q"}


class TestAgentToolBoxRagPolicy:
    def test_list_tools_hides_override_params_when_router_off(
        self,
        rag_search_schema: dict,
    ) -> None:
        inner = MagicMock(spec=ToolBox)
        inner.list_tools.return_value = [rag_search_schema]
        box = AgentToolBox(inner=inner, enable_rag_profile_router=False)

        schemas = box.list_tools()
        props = schemas[0]["function"]["parameters"]["properties"]
        assert set(props) == {RAG_SEARCH_QUERY_ARG}

    def test_list_tools_exposes_override_params_when_router_on(
        self,
        rag_search_schema: dict,
    ) -> None:
        inner = MagicMock(spec=ToolBox)
        inner.list_tools.return_value = [rag_search_schema]
        box = AgentToolBox(inner=inner, enable_rag_profile_router=True)

        schemas = box.list_tools()
        props = schemas[0]["function"]["parameters"]["properties"]
        assert USE_HYDE_KEY in props

    @pytest.mark.asyncio
    async def test_ainvoke_strips_override_args_when_router_off(self) -> None:
        inner = MagicMock(spec=ToolBox)
        inner.ainvoke = AsyncMock(return_value=MagicMock(output="ok", error=None))
        box = AgentToolBox(inner=inner, enable_rag_profile_router=False)

        await box.ainvoke(
            DEFAULT_RAG_TOOL_NAME,
            {"query": "q", USE_HYDE_KEY: True},
        )

        inner.ainvoke.assert_awaited_once_with(DEFAULT_RAG_TOOL_NAME, {"query": "q"})

    @pytest.mark.asyncio
    async def test_ainvoke_passes_override_args_when_router_on(self) -> None:
        inner = MagicMock(spec=ToolBox)
        inner.ainvoke = AsyncMock(return_value=MagicMock(output="ok", error=None))
        box = AgentToolBox(inner=inner, enable_rag_profile_router=True)
        args = {"query": "q", USE_HYDE_KEY: True}

        await box.ainvoke(DEFAULT_RAG_TOOL_NAME, args)

        inner.ainvoke.assert_awaited_once_with(DEFAULT_RAG_TOOL_NAME, args)
