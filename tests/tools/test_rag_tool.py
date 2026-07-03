"""Verify RAG tools and :mod:`rag.context` binding via :class:`ToolContextBundle`."""

from __future__ import annotations

import pytest

from rag.context import (
    RAG_CONTEXT_KEY,
    RAG_INDEX_PROFILE_KEY,
    RAG_SEARCH_PROFILE_KEY,
    RagToolContext,
    bind_rag,
    build_rag_context,
)
from rag.profile_schema import (
    RECALL_N_KEY,
    TOP_K_KEY,
    USE_CONTEXTUAL_KEY,
    USE_HYDE_KEY,
    USE_PREDICT_QUESTIONS_KEY,
    USE_RERANKER_KEY,
    USE_SMALL_TO_BIG_KEY,
    USE_TOKEN_CHUNKER_KEY,
    default_search_profile,
    normalize_search_profile,
)
from tools.LocalTool import RAG_tool
from tools.context import INJECTED_CONTEXT_PARAM, ToolContextBundle
from tools.registry import TOOL_INFO_ATTR
from tools.tool_box import ToolBox

RAG_TOOL_PACKAGE = ("tools.LocalTool.RAG_tool",)


def _rag_tool_box(bundle: ToolContextBundle) -> ToolBox:
    return ToolBox(autodiscover=True, packages=RAG_TOOL_PACKAGE, context=bundle)


@pytest.fixture
def rag_tool_box():
    return ToolBox(autodiscover=True, packages=RAG_TOOL_PACKAGE)


@pytest.fixture
def full_rag_bundle(mock_embedder, in_memory_vector_store):
    bundle = ToolContextBundle()
    bind_rag(
        bundle,
        collection="full_ctx",
        in_memory=True,
        store=in_memory_vector_store,
        embedder=mock_embedder,
        index_profile_id="baseline",
        retrieve_profile_id="rerank_contextual",
    )
    return bundle


@pytest.fixture
def full_rag_tool_box(full_rag_bundle):
    return _rag_tool_box(full_rag_bundle)


@pytest.fixture
def bound_rag_tool_box(mock_embedder, in_memory_vector_store):
    bundle = ToolContextBundle()
    bind_rag(
        bundle,
        collection="bound_ctx",
        in_memory=True,
        store=in_memory_vector_store,
        embedder=mock_embedder,
    )
    return _rag_tool_box(bundle)


def _tool_param_names(rag_tool_box: ToolBox, tool_name: str) -> set[str]:
    schemas = rag_tool_box.list_tools()
    schema = next(s for s in schemas if s["function"]["name"] == tool_name)
    props = schema["function"]["parameters"].get("properties", {})
    return set(props.keys())


@pytest.mark.unit
def test_rag_tools_are_async():
    import asyncio

    assert asyncio.iscoroutinefunction(RAG_tool.RAG_index_tool)
    assert asyncio.iscoroutinefunction(RAG_tool.RAG_search_tool)


@pytest.mark.unit
def test_rag_tools_have_decorator_metadata():
    index_info = getattr(RAG_tool.RAG_index_tool, TOOL_INFO_ATTR)
    search_info = getattr(RAG_tool.RAG_search_tool, TOOL_INFO_ATTR)

    assert index_info.name == "RAG_index_tool"
    assert search_info.name == "RAG_search_tool"
    assert index_info.context_keys == (RAG_CONTEXT_KEY, RAG_INDEX_PROFILE_KEY)
    assert search_info.context_keys == (RAG_CONTEXT_KEY, RAG_SEARCH_PROFILE_KEY)
    assert index_info.tool_path == "tools.LocalTool.RAG_tool.RAG_index_tool"
    assert search_info.tool_path == "tools.LocalTool.RAG_tool.RAG_search_tool"


@pytest.mark.unit
def test_rag_tools_appear_in_isolated_tool_box(rag_tool_box):
    names = {schema["function"]["name"] for schema in rag_tool_box.list_tools()}
    assert names == {"RAG_index_tool", "RAG_search_tool"}


@pytest.mark.unit
def test_search_tool_schema_hides_injected_context(rag_tool_box):
    params = _tool_param_names(rag_tool_box, "RAG_search_tool")
    assert INJECTED_CONTEXT_PARAM not in params
    assert params >= {
        "query",
        USE_CONTEXTUAL_KEY,
        USE_SMALL_TO_BIG_KEY,
        USE_HYDE_KEY,
        USE_RERANKER_KEY,
        RECALL_N_KEY,
        TOP_K_KEY,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_index_tool_requires_context(rag_tool_box):
    result = await rag_tool_box.ainvoke(
        "RAG_index_tool",
        {"text": "hello", "source": "doc.md"},
    )
    assert "Missing tool context" in str(result.error)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_search_tool_requires_context(rag_tool_box):
    result = await rag_tool_box.ainvoke("RAG_search_tool", {"query": "query"})
    assert "Missing tool context" in str(result.error)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_index_tool_indexes_document(bound_rag_tool_box):
    result = await bound_rag_tool_box.ainvoke(
        "RAG_index_tool",
        {
            "text": "# Note\n\nunique rag tool indexing phrase",
            "source": "rag_tool_test.md",
        },
    )
    assert result.error is None
    assert result.output == "Successfully indexed document 'rag_tool_test.md'."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_search_tool_returns_indexed_content(bound_rag_tool_box):
    await bound_rag_tool_box.ainvoke(
        "RAG_index_tool",
        {
            "text": "# Note\n\nobservability dashboards and metrics",
            "source": "obs.md",
        },
    )
    result = await bound_rag_tool_box.ainvoke(
        "RAG_search_tool",
        {"query": "observability dashboards"},
    )
    assert result.error is None
    assert "observability dashboards" in str(result.output)


class TestRagToolContext:
    @pytest.mark.unit
    def test_build_rag_context_is_connections_only(
        self, mock_embedder, in_memory_vector_store
    ) -> None:
        ctx = build_rag_context(
            collection="conn_only",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
        )
        assert ctx.collection == "conn_only"
        assert ctx.store is in_memory_vector_store
        assert ctx.embedder is mock_embedder
        assert not hasattr(ctx, "default_search_profile")

    @pytest.mark.unit
    def test_bind_rag_registers_connections_and_profiles(
        self, mock_embedder, in_memory_vector_store
    ) -> None:
        bundle = ToolContextBundle()
        ctx = bind_rag(
            bundle,
            collection="bundle_test",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
            index_profile_id="baseline",
            retrieve_profile_id="rerank_contextual",
        )
        assert bundle.require(RAG_CONTEXT_KEY, RagToolContext) is ctx
        profile = bundle.require(RAG_SEARCH_PROFILE_KEY, dict)
        assert profile[USE_CONTEXTUAL_KEY] is True
        assert profile[USE_RERANKER_KEY] is True


class TestProfileDefaults:
    @pytest.mark.unit
    def test_search_profile_merge_from_bundle_defaults(self, full_rag_bundle) -> None:
        defaults = full_rag_bundle.require(RAG_SEARCH_PROFILE_KEY, dict)
        eff, _ = normalize_search_profile(
            None,
            defaults=defaults,
            max_recall_n=50,
            max_top_k=None,
        )
        assert eff[USE_CONTEXTUAL_KEY] is True
        assert eff[USE_RERANKER_KEY] is True
        assert eff[USE_HYDE_KEY] is False

    @pytest.mark.unit
    def test_partial_override_keeps_other_defaults(self, full_rag_bundle) -> None:
        defaults = full_rag_bundle.require(RAG_SEARCH_PROFILE_KEY, dict)
        eff, _ = normalize_search_profile(
            {USE_HYDE_KEY: True},
            defaults=defaults,
            max_recall_n=50,
            max_top_k=None,
        )
        assert eff[USE_HYDE_KEY] is True
        assert eff[USE_CONTEXTUAL_KEY] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_context_index_and_search_with_explicit_flags(full_rag_tool_box):
    index_result = await full_rag_tool_box.ainvoke(
        "RAG_index_tool",
        {
            "text": "# Note\n\npipeline profile override phrase",
            "source": "profile.md",
            "use_token_chunker": False,
            "use_contextual": False,
            "use_small_to_big": False,
            "use_predict_questions": False,
        },
    )
    assert index_result.error is None
    assert "Successfully indexed" in str(index_result.output)

    search_result = await full_rag_tool_box.ainvoke(
        "RAG_search_tool",
        {
            "query": "pipeline profile override",
            "use_contextual": False,
            "use_small_to_big": False,
            "use_hyde": False,
            "use_reranker": False,
            "top_k": 2,
        },
    )
    assert search_result.error is None
    assert "pipeline profile override" in str(search_result.output)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_tools_via_tool_box(bound_rag_tool_box):
    index_result = await bound_rag_tool_box.ainvoke(
        "RAG_index_tool",
        {
            "text": "# Doc\n\nlanggraph agent retrieval smoke test",
            "source": "agent.md",
        },
    )
    assert index_result.error is None
    assert "Successfully indexed" in str(index_result.output)

    search_result = await bound_rag_tool_box.ainvoke(
        "RAG_search_tool",
        {"query": "langgraph agent retrieval"},
    )
    assert search_result.error is None
    assert "langgraph agent retrieval" in str(search_result.output)
