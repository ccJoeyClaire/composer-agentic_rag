"""Retrieval gate capability configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent_v2.core.state import AgentState


ScorePassagesFn = Callable[[AgentState, str, list[str]], list[float]]


@dataclass
class RetrievalGateConfig:
    """Settings for the retrieval quality gate (stub scoring hooks)."""

    score_fn: ScorePassagesFn | None = None
    pass_threshold: float = 0.5
