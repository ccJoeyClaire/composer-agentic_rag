"""Parity tests for duplicated indexing helpers in ``rag_eval`` vs ``agent_eval``.

``rag_eval.pipeline`` and ``agent_eval.pipeline`` each own their own copy of
``drop_collection`` and ``index_doc_list`` so the two eval lines stay independent.
When you change one copy, update the other or adjust these tests deliberately.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from _eval_.agent_eval.pipeline import drop_collection as agent_drop_collection
from _eval_.agent_eval.pipeline import index_doc_list as agent_index_doc_list
from _eval_.data_preparing.beir import CorpusDoc
from _eval_.rag_eval.pipeline import drop_collection as rag_drop_collection
from _eval_.rag_eval.pipeline import index_doc_list as rag_index_doc_list
from rag.build import build_RAG_indexer

_INDEX_DOC_LIST_IMPLS = [rag_index_doc_list, agent_index_doc_list]
_DROP_COLLECTION_IMPLS = [rag_drop_collection, agent_drop_collection]


@pytest.mark.asyncio
@pytest.mark.parametrize("index_doc_list", _INDEX_DOC_LIST_IMPLS)
async def test_index_doc_list_returns_doc_count(
    mock_embedder,
    in_memory_qdrant_store,
    index_doc_list,
) -> None:
    indexer = build_RAG_indexer(
        "test_index_doc_list",
        in_memory=True,
        store=in_memory_qdrant_store,
        embedder=mock_embedder,
    )
    docs = [
        CorpusDoc(doc_id="d1", text="first document about biology."),
        CorpusDoc(doc_id="d2", text="second document about chemistry."),
    ]
    count = await index_doc_list(indexer, docs, concurrency=2)
    assert count == len(docs)


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_collection", _DROP_COLLECTION_IMPLS)
async def test_drop_collection_deletes_when_exists(drop_collection) -> None:
    store = AsyncMock()
    store.collection = "test_coll"
    store.client.collection_exists.return_value = True

    await drop_collection(store)

    store.client.collection_exists.assert_awaited_once_with("test_coll")
    store.client.delete_collection.assert_awaited_once_with("test_coll")


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_collection", _DROP_COLLECTION_IMPLS)
async def test_drop_collection_noop_when_missing(drop_collection) -> None:
    store = AsyncMock()
    store.collection = "test_coll"
    store.client.collection_exists.return_value = False

    await drop_collection(store)

    store.client.collection_exists.assert_awaited_once_with("test_coll")
    store.client.delete_collection.assert_not_called()
