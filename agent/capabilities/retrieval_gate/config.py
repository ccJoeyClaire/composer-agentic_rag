"""Retrieval gate capability configuration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent.core.state import AgentState
from rag.reranker.dashscope_rerank_client import DashScopeRerankClient

ScorePassagesFn = Callable[
    [AgentState, str, list[str]],
    Awaitable[list[float]],
]

# qwen3-rerank relevance_score is 0~1. Official examples: clearly relevant ~0.7+,
# weak/tangential ~0.3–0.4. 0.5 requires at least moderately relevant top hit.
DEFAULT_PASS_THRESHOLD = 0.5


DEFAULT_MAX_SCORING_RETRIES = 2


@dataclass
class RetrievalGateConfig:
    """Settings for the retrieval quality gate."""

    score_fn: ScorePassagesFn | None = None
    rerank_client: DashScopeRerankClient | None = None
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
    max_scoring_retries: int = DEFAULT_MAX_SCORING_RETRIES
