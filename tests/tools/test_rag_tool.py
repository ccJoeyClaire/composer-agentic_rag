"""Verify tools.LocalTool.RAG_tool: bind_rag_context + legacy bind_indexer / bind_retriever."""

from __future__ import annotations

import pytest

from rag.chunker.semantic_chunker import SemanticChunker
from rag.core import RAGIndexer, RAGRetriever
from rag.retriever.vector_retriever import VectorRetriever
from tools.LocalTool import RAG_tool
from tools.registry import TOOL_INFO_ATTR
from tools.tool_box import ToolBox

RAG_TOOL_PACKAGE = ("tools.LocalTool.RAG_tool",)


@pytest.fixture
def reset_rag_bindings():
    RAG_tool._context = RAG_tool.RagToolContext()
    yield
    RAG_tool._context = RAG_tool.RagToolContext()


@pytest.fixture
def rag_tool_box():
    return ToolBox(autodiscover=True, packages=RAG_TOOL_PACKAGE)


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
    RAG_tool.bind_indexer(indexer)
    RAG_tool.bind_retriever(retriever, top_k=2)
    return indexer, retriever


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


@pytest.mark.unit
def test_full_context_caches_and_clamps_retriever_variants(
    mock_embedder, in_memory_vector_store, reset_rag_bindings
):
    RAG_tool.bind_rag_context(
        collection="variant_test",
        in_memory=True,
        store=in_memory_vector_store,
        embedder=mock_embedder,
        max_recall_n=10,
    )
    ctx = RAG_tool._context

    r1, _ = ctx.resolve_retriever(use_hyde=False, use_reranker=False, recall_n=5)
    r2, _ = ctx.resolve_retriever(use_hyde=False, use_reranker=False, recall_n=5)
    assert r1 is r2  # same key -> cached

    r3, _ = ctx.resolve_retriever(use_hyde=False, use_reranker=False, recall_n=7)
    assert r3 is not r1  # different recall_n -> new variant

    r4, notes = ctx.resolve_retriever(use_hyde=False, use_reranker=False, recall_n=999)
    assert any("clamp" in note for note in notes)
    assert r4.recall_n == 10


@pytest.mark.unit
def test_disabled_query_option_emits_note(
    mock_embedder, in_memory_vector_store, reset_rag_bindings
):
    RAG_tool.bind_rag_context(
        collection="disabled_test",
        in_memory=True,
        store=in_memory_vector_store,
        embedder=mock_embedder,
        allow_hyde=False,
    )
    _, notes = RAG_tool._context.resolve_retriever(
        use_hyde=True, use_reranker=False, recall_n=None
    )
    assert any("use_hyde" in note for note in notes)


@pytest.mark.unit
def test_disabled_predict_questions_emits_note(
    mock_embedder, in_memory_vector_store, reset_rag_bindings
):
    RAG_tool.bind_rag_context(
        collection="predict_test",
        in_memory=True,
        store=in_memory_vector_store,
        embedder=mock_embedder,
        allow_predict_questions=False,
    )
    indexer, notes = RAG_tool._context.resolve_indexer(use_predict_questions=True)
    assert indexer is not None
    assert any("use_predict_questions" in note for note in notes)


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


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/tools/test_rag_tool.py -v
# ================================================================================================================
