"""Direct RAG + LLM infer for ``EvalRunner(mode='rag')``."""

from __future__ import annotations

from eval.base import EvalRunner, GoldSample, RagInferArtifact


async def run_rag_infer(
    runner: EvalRunner,
    gold: list[GoldSample],
) -> list[RagInferArtifact]:
    """Retrieve with ``gold_query``, generate with ``user_question`` + context.

    TODO: ``aquery_trace`` like ``get_start.retrieve_example`` (``final`` stage);
    call ``LLMClient`` with user_question and retrieved chunks.
    """
    raise NotImplementedError
