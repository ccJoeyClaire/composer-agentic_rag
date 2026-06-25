"""Abstract base for inference arms.

Both :class:`~eval.infer.direct_rag.DirectRagArm` and
:class:`~eval.infer.agent.AgentInferArm` implement this contract so the
assembler and batch runner can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from eval.run_config import EvalRunConfig
from eval.types import InferResult


class BaseInferArm(ABC):
    """Run one query through a retrieval/generation pipeline.

    Each concrete subclass encapsulates one *arm* of the eval matrix:
    either the direct-RAG retrieve+generate path or an agent pattern.
    """

    def __init__(self, config: EvalRunConfig) -> None:
        self._config = config

    @property
    def config(self) -> EvalRunConfig:
        return self._config

    @abstractmethod
    async def arun(self, query_id: str, query: str) -> InferResult:
        """Execute one query and return a fully populated :class:`InferResult`.

        Args:
            query_id: Stable gold record identifier.
            query:    Natural-language question string.

        Returns:
            :class:`InferResult` with response text and retrieved context.
        """
        ...

    async def arun_batch(
        self, records: list[tuple[str, str]]
    ) -> list[InferResult]:
        """Run multiple (query_id, query) pairs sequentially.

        Subclasses may override with a concurrent implementation when the
        underlying API supports parallel calls.

        Args:
            records: List of ``(query_id, query)`` pairs.

        Returns:
            :class:`InferResult` for each record in the same order.
        """
        results: list[InferResult] = []
        for query_id, query in records:
            results.append(await self.arun(query_id, query))
        return results

    async def aclose(self) -> None:
        """Release resources (e.g. Qdrant client connection).

        Default is a no-op; subclasses that hold open connections should override.
        """
