"""DirectRagArm: retrieve → single-turn LLM generate.

This is the "直接 RAG" baseline defined in blueprint §4.3:
the same retriever pipeline used in ``get_start/retrieve_example.py``,
followed by a single LLM call (no agent graph, no reflection).
"""

from __future__ import annotations

from typing import Any

from llm.client import LLMClient
from rag.base import Chunk
from rag.build import build_RAG_retriever
from rag.config import get_profile, get_rag_config

from eval.infer.base import BaseInferArm
from eval.run_config import EvalRunConfig
from eval.types import ContextChunk, InferResult

_SYSTEM_PROMPT = (
    "你是一个知识问答助手。请仅根据下方提供的参考段落回答问题。"
    "如果参考内容不足以回答，请如实说明。"
)

_CONTEXT_TEMPLATE = "参考段落：\n{context}\n\n问题：{query}"


def _chunk_to_context(chunk: Chunk) -> ContextChunk:
    """Convert a :class:`~rag.base.Chunk` to RAGChecker's context format.

    Uses raw ``content`` (no contextual header) per blueprint §4.2.
    ``doc_id`` prefers ``chunk_id``; falls back to ``source``.
    """
    meta = chunk.metadata
    doc_id: str = str(meta.get("chunk_id") or meta.get("source") or "")
    return ContextChunk(doc_id=doc_id, text=chunk.content)


def _build_prompt(query: str, chunks: list[Chunk]) -> list[dict[str, Any]]:
    context_text = "\n\n---\n\n".join(c.content for c in chunks)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _CONTEXT_TEMPLATE.format(context=context_text, query=query),
        },
    ]


class DirectRagArm(BaseInferArm):
    """Retrieve with the profile's RAGRetriever, then generate in one LLM call.

    Args:
        config: Run config supplying profile_id and collection name.
    """

    def __init__(self, config: EvalRunConfig) -> None:
        super().__init__(config)
        rag_config = get_rag_config()
        profile = get_profile(rag_config, config.profile_id)
        self._top_k: int = rag_config.retriever.top_k
        self._retriever = build_RAG_retriever(
            config.collection,
            use_reranker=profile.use_reranker,
            use_contextual=profile.use_contextual,
            use_hyde=profile.use_hyde,
            use_small_to_big=profile.use_small_to_big,
        )
        self._llm = LLMClient()

    async def arun(self, query_id: str, query: str) -> InferResult:
        """Retrieve relevant chunks then generate a grounded answer.

        Args:
            query_id: Stable gold record identifier.
            query:    Natural-language question string.

        Returns:
            :class:`InferResult` with generated response and retrieved chunks.
        """
        result = await self._retriever.aquery_trace(query, top_k=self._top_k)
        chunks = result.chunks

        messages = _build_prompt(query, chunks)
        llm_msg = await self._llm.arequest_llm(messages)
        response: str = str(llm_msg.content or "")

        return InferResult(
            query_id=query_id,
            query=query,
            response=response,
            retrieved_context=[_chunk_to_context(c) for c in chunks],
            arm=self._config.arm,
            profile_id=self._config.profile_id,
        )

    async def aclose(self) -> None:
        """Close the underlying Qdrant connection."""
        chain = self._retriever.retriever
        store = getattr(chain, "store", None) or getattr(
            getattr(chain, "inner", None), "store", None
        )
        if store is not None:
            await store.aclose()
