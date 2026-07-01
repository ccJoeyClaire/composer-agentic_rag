"""Verify RAG tools and :mod:`rag.context` binding (full + legacy paths)."""

from __future__ import annotations

import pytest

from rag.chunker.semantic_chunker import SemanticChunker
from rag.context import (
    bind_indexer,
    bind_rag_context,
    bind_retriever,
    get_active_context,
    reset_rag_context,
)
from rag.core import RAGIndexer, RAGRetriever
from rag.profile_schema import (
    RECALL_N_KEY,
    TOP_K_KEY,
    USE_CONTEXTUAL_KEY,
    USE_HYDE_KEY,
    USE_PREDICT_QUESTIONS_KEY,
    USE_RERANKER_KEY,
    USE_SMALL_TO_BIG_KEY,
    USE_TOKEN_CHUNKER_KEY,
)
from rag.retriever.vector_retriever import VectorRetriever
from tools.LocalTool import RAG_tool
from tools.registry import TOOL_INFO_ATTR
from tools.tool_box import ToolBox

RAG_TOOL_PACKAGE = ("tools.LocalTool.RAG_tool",)

_SEARCH_BOOL_OFF = {
    USE_CONTEXTUAL_KEY: False,
    USE_SMALL_TO_BIG_KEY: False,
    USE_HYDE_KEY: False,
    USE_RERANKER_KEY: False,
}


@pytest.fixture
def reset_rag_bindings():
    reset_rag_context()
    yield
    reset_rag_context()


@pytest.fixture
def rag_tool_box():
    return ToolBox(autodiscover=True, packages=RAG_TOOL_PACKAGE)


@pytest.fixture
def full_rag_context(mock_embedder, in_memory_vector_store, reset_rag_bindings):
    bind_rag_context(
        collection="full_ctx",
        in_memory=True,
        store=in_memory_vector_store,
        embedder=mock_embedder,
        index_profile_id="baseline",
        retrieve_profile_id="rerank_contextual",
        max_recall_n=50,
    )
    return get_active_context()


@pytest.fixture
def bound_rag_stack(
    mock_embedder, mock_chunker_embedding_client, in_memory_vector_store, reset_rag_bindings
):
    indexer = RAGIndexer(
        chunker=SemanticChunker(
            chunk_tokens=256,
            overlap_tokens=0,
            min_chunk_tokens=1,
            embedding_client=mock_chunker_embedding_client,
        ),
        embedder=mock_embedder,
        store=in_memory_vector_store,
    )
    retriever = RAGRetriever(
        retriever=VectorRetriever(
            embedder=mock_embedder,
            store=in_memory_vector_store,
        ),
        recall_n=5,
    )
    bind_indexer(indexer)
    bind_retriever(retriever, top_k=2)
    return indexer, retriever


def _tool_param_names(rag_tool_box: ToolBox, tool_name: str) -> set[str]:
    schemas = rag_tool_box.list_tools()
    schema = next(s for s in schemas if s["function"]["name"] == tool_name)
    props = schema["function"]["parameters"].get("properties", {})
    return set(props.keys())


@pytest.mark.unit
def test_rag_tools_have_decorator_metadata():
    index_info = getattr(RAG_tool.RAG_index_tool, TOOL_INFO_ATTR)
    search_info = getattr(RAG_tool.RAG_search_tool, TOOL_INFO_ATTR)

    assert index_info.name == "RAG_index_tool"
    assert search_info.name == "RAG_search_tool"
    assert index_info.tool_path == "tools.LocalTool.RAG_tool.RAG_index_tool"
    assert search_info.tool_path == "tools.LocalTool.RAG_tool.RAG_search_tool"
    assert index_info.source == search_info.source == "local"


@pytest.mark.unit
def test_rag_tools_appear_in_isolated_tool_box(rag_tool_box):
    names = {schema["function"]["name"] for schema in rag_tool_box.list_tools()}
    assert names == {"RAG_index_tool", "RAG_search_tool"}


@pytest.mark.unit
def test_search_tool_schema_exposes_full_search_profile(rag_tool_box):
    params = _tool_param_names(rag_tool_box, "RAG_search_tool")
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
def test_index_tool_schema_exposes_full_index_profile(rag_tool_box):
    params = _tool_param_names(rag_tool_box, "RAG_index_tool")
    assert params >= {
        "text",
        "source",
        USE_TOKEN_CHUNKER_KEY,
        USE_CONTEXTUAL_KEY,
        USE_SMALL_TO_BIG_KEY,
        USE_PREDICT_QUESTIONS_KEY,
    }


@pytest.mark.unit
def test_rag_index_tool_requires_bind(reset_rag_bindings):
    result = RAG_tool.RAG_index_tool("hello", source="doc.md")
    assert "bind_indexer" in result


@pytest.mark.unit
def test_rag_search_tool_requires_bind(reset_rag_bindings):
    result = RAG_tool.RAG_search_tool("query")
    assert "bind_retriever" in result


@pytest.mark.integration
def test_rag_index_tool_indexes_document(bound_rag_stack):
    result = RAG_tool.RAG_index_tool(
        "# Note\n\nunique rag tool indexing phrase",
        source="rag_tool_test.md",
    )
    assert result == "Successfully indexed document 'rag_tool_test.md'."


@pytest.mark.integration
def test_rag_search_tool_returns_indexed_content(bound_rag_stack):
    RAG_tool.RAG_index_tool(
        "# Note\n\nobservability dashboards and metrics",
        source="obs.md",
    )
    result = RAG_tool.RAG_search_tool("observability dashboards")
    assert "observability dashboards" in result


@pytest.mark.unit
def test_legacy_fixed_retriever_notes_on_options(bound_rag_stack):
    result = RAG_tool.RAG_search_tool("observability dashboards", use_hyde=True)
    assert "[note]" in result
    assert "use_hyde" in result


class TestRetrieverCacheKeys:
    @pytest.mark.unit
    def test_same_full_profile_hits_cache(self, full_rag_context) -> None:
        profile = {**_SEARCH_BOOL_OFF, RECALL_N_KEY: 5}
        r1, _, _ = full_rag_context.resolve_retriever(profile)
        r2, _, _ = full_rag_context.resolve_retriever(profile)
        assert r1 is r2

    @pytest.mark.unit
    def test_recall_n_splits_cache(self, full_rag_context) -> None:
        base = {**_SEARCH_BOOL_OFF}
        r1, _, _ = full_rag_context.resolve_retriever({**base, RECALL_N_KEY: 5})
        r2, _, _ = full_rag_context.resolve_retriever({**base, RECALL_N_KEY: 7})
        assert r1 is not r2

    @pytest.mark.unit
    def test_use_contextual_splits_cache(self, full_rag_context) -> None:
        base = {**_SEARCH_BOOL_OFF, RECALL_N_KEY: 5}
        r_off, _, _ = full_rag_context.resolve_retriever(
            {**base, USE_CONTEXTUAL_KEY: False}
        )
        r_on, _, _ = full_rag_context.resolve_retriever(
            {**base, USE_CONTEXTUAL_KEY: True}
        )
        assert r_off is not r_on
        assert r_off.contextual_enricher is None
        assert r_on.contextual_enricher is not None

    @pytest.mark.unit
    def test_use_small_to_big_splits_cache(self, full_rag_context) -> None:
        base = {**_SEARCH_BOOL_OFF, RECALL_N_KEY: 5}
        r_off, _, _ = full_rag_context.resolve_retriever(
            {**base, USE_SMALL_TO_BIG_KEY: False}
        )
        r_on, _, _ = full_rag_context.resolve_retriever(
            {**base, USE_SMALL_TO_BIG_KEY: True}
        )
        assert r_off is not r_on

    @pytest.mark.unit
    def test_use_hyde_splits_cache(self, full_rag_context) -> None:
        base = {**_SEARCH_BOOL_OFF, RECALL_N_KEY: 5}
        r_off, _, _ = full_rag_context.resolve_retriever({**base, USE_HYDE_KEY: False})
        r_on, _, _ = full_rag_context.resolve_retriever({**base, USE_HYDE_KEY: True})
        assert r_off is not r_on
        assert r_off.query_transformer is None
        assert r_on.query_transformer is not None

    @pytest.mark.unit
    def test_clamps_recall_n(self, full_rag_context) -> None:
        profile = {**_SEARCH_BOOL_OFF, RECALL_N_KEY: 999}
        full_rag_context.max_recall_n = 10
        _, notes, eff = full_rag_context.resolve_retriever(profile)
        assert any("clamp" in note for note in notes)
        assert eff[RECALL_N_KEY] == 10


class TestIndexerCacheKeys:
    @pytest.mark.unit
    def test_use_token_chunker_splits_cache(self, full_rag_context) -> None:
        base = {
            USE_TOKEN_CHUNKER_KEY: False,
            USE_CONTEXTUAL_KEY: False,
            USE_SMALL_TO_BIG_KEY: False,
            USE_PREDICT_QUESTIONS_KEY: False,
        }
        i_sem, _, _ = full_rag_context.resolve_indexer(base)
        i_tok, _, _ = full_rag_context.resolve_indexer(
            {**base, USE_TOKEN_CHUNKER_KEY: True}
        )
        assert i_sem is not i_tok

    @pytest.mark.unit
    def test_use_predict_questions_splits_cache(self, full_rag_context) -> None:
        base = {
            USE_TOKEN_CHUNKER_KEY: False,
            USE_CONTEXTUAL_KEY: False,
            USE_SMALL_TO_BIG_KEY: False,
            USE_PREDICT_QUESTIONS_KEY: False,
        }
        i_off, _, _ = full_rag_context.resolve_indexer(base)
        i_on, _, _ = full_rag_context.resolve_indexer(
            {**base, USE_PREDICT_QUESTIONS_KEY: True}
        )
        assert i_off is not i_on


class TestProfileDefaultsAndGates:
    @pytest.mark.unit
    def test_omitted_search_profile_merges_baseline(self, full_rag_context) -> None:
        _, _, eff = full_rag_context.resolve_retriever(None)
        assert eff[USE_CONTEXTUAL_KEY] is True
        assert eff[USE_RERANKER_KEY] is True
        assert eff[USE_HYDE_KEY] is False
        assert eff[USE_SMALL_TO_BIG_KEY] is False

    @pytest.mark.unit
    def test_partial_override_keeps_other_defaults(self, full_rag_context) -> None:
        _, _, eff = full_rag_context.resolve_retriever({USE_HYDE_KEY: True})
        assert eff[USE_HYDE_KEY] is True
        assert eff[USE_CONTEXTUAL_KEY] is True
        assert eff[USE_RERANKER_KEY] is True

    @pytest.mark.unit
    def test_top_k_override_and_clamp(
        self, mock_embedder, in_memory_vector_store, reset_rag_bindings
    ) -> None:
        bind_rag_context(
            collection="topk_test",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
            max_top_k=2,
        )
        _, notes, eff = get_active_context().resolve_retriever({TOP_K_KEY: 5})
        assert any("top_k" in note for note in notes)
        assert eff[TOP_K_KEY] == 2

    @pytest.mark.unit
    def test_disabled_contextual_emits_note(
        self, mock_embedder, in_memory_vector_store, reset_rag_bindings
    ) -> None:
        bind_rag_context(
            collection="ctx_gate",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
            allow_contextual=False,
        )
        _, notes, eff = get_active_context().resolve_retriever({USE_CONTEXTUAL_KEY: True})
        assert any("use_contextual" in note for note in notes)
        assert eff[USE_CONTEXTUAL_KEY] is False

    @pytest.mark.unit
    def test_disabled_small_to_big_emits_note(
        self, mock_embedder, in_memory_vector_store, reset_rag_bindings
    ) -> None:
        bind_rag_context(
            collection="s2b_gate",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
            allow_small_to_big=False,
        )
        _, notes, eff = get_active_context().resolve_retriever({USE_SMALL_TO_BIG_KEY: True})
        assert any("use_small_to_big" in note for note in notes)
        assert eff[USE_SMALL_TO_BIG_KEY] is False

    @pytest.mark.unit
    def test_disabled_hyde_emits_note(
        self, mock_embedder, in_memory_vector_store, reset_rag_bindings
    ) -> None:
        bind_rag_context(
            collection="disabled_test",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
            allow_hyde=False,
        )
        _, notes, eff = get_active_context().resolve_retriever({USE_HYDE_KEY: True})
        assert any("use_hyde" in note for note in notes)
        assert eff[USE_HYDE_KEY] is False

    @pytest.mark.unit
    def test_disabled_predict_questions_emits_note(
        self, mock_embedder, in_memory_vector_store, reset_rag_bindings
    ) -> None:
        bind_rag_context(
            collection="predict_test",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
            allow_predict_questions=False,
        )
        _, notes, _ = get_active_context().resolve_indexer({USE_PREDICT_QUESTIONS_KEY: True})
        assert any("use_predict_questions" in note for note in notes)

    @pytest.mark.unit
    def test_bind_rag_context_seeds_baseline_defaults(
        self, mock_embedder, in_memory_vector_store, reset_rag_bindings
    ) -> None:
        ctx = bind_rag_context(
            collection="profile_test",
            in_memory=True,
            store=in_memory_vector_store,
            embedder=mock_embedder,
            index_profile_id="baseline",
            retrieve_profile_id="rerank_contextual",
        )
        assert ctx.default_search_profile[USE_SMALL_TO_BIG_KEY] is False
        assert ctx.default_search_profile[USE_CONTEXTUAL_KEY] is True
        assert ctx.default_search_profile[USE_HYDE_KEY] is False
        assert ctx.default_search_profile[USE_RERANKER_KEY] is True
        assert ctx.default_index_profile[USE_CONTEXTUAL_KEY] is True


@pytest.mark.integration
def test_full_context_index_and_search_with_explicit_flags(
    full_rag_context,
):
    """Index + search via tools; disable rerank/hyde to keep test lightweight."""
    index_result = RAG_tool.RAG_index_tool(
        "# Note\n\npipeline profile override phrase",
        source="profile.md",
        use_token_chunker=False,
        use_contextual=False,
        use_small_to_big=False,
        use_predict_questions=False,
    )
    assert "Successfully indexed" in index_result

    search_result = RAG_tool.RAG_search_tool(
        "pipeline profile override",
        use_contextual=False,
        use_small_to_big=False,
        use_hyde=False,
        use_reranker=False,
        top_k=2,
    )
    assert "pipeline profile override" in search_result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_tools_via_tool_box(rag_tool_box, bound_rag_stack):
    index_result = await rag_tool_box.ainvoke(
        "RAG_index_tool",
        {
            "text": "# Doc\n\nlanggraph agent retrieval smoke test",
            "source": "agent.md",
        },
    )
    assert index_result.error is None
    assert "Successfully indexed" in str(index_result.output)

    search_result = await rag_tool_box.ainvoke(
        "RAG_search_tool",
        {"query": "langgraph agent retrieval"},
    )
    assert search_result.error is None
    assert "langgraph agent retrieval" in str(search_result.output)
