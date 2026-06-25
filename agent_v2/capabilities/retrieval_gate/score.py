"""Default passage scoring via DashScope rerank."""

from __future__ import annotations

from agent_v2.capabilities.retrieval_gate.config import ScorePassagesFn
from agent_v2.core.state import AgentState
from rag.reranker.dashscope_rerank_client import DashScopeRerankClient


def build_dashscope_score_fn(
    client: DashScopeRerankClient | None = None,
) -> ScorePassagesFn:
    """Return an async scorer that calls ``DashScopeRerankClient.ascore_documents``."""
    rerank_client = client or DashScopeRerankClient()

    async def score(
        state: AgentState,
        query: str,
        passages: list[str],
    ) -> list[float]:
        _ = state
        return await rerank_client.ascore_documents(query, passages)

    return score
