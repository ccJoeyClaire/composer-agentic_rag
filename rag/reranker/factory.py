"""Select a rerank backend from environment configuration."""

from __future__ import annotations

import os
from typing import Literal

from ..base import BaseReranker
from .cross_encoder_reranker import CrossEncoderReranker
from .dashscope_reranker import DashScopeReranker

RerankBackend = Literal["auto", "dashscope", "cross_encoder"]


def resolve_rerank_backend() -> Literal["dashscope", "cross_encoder"]:
    """Pick rerank implementation for ``auto`` / explicit ``RERANK_BACKEND``."""
    explicit = (os.environ.get("RERANK_BACKEND") or "auto").lower()
    if explicit == "dashscope":
        return "dashscope"
    if explicit == "cross_encoder":
        return "cross_encoder"
    if os.environ.get("RERANK_MODEL_ID"):
        return "dashscope"
    return "cross_encoder"


def make_reranker(*, enabled: bool) -> BaseReranker | None:
    """Build the configured reranker, or ``None`` when reranking is disabled."""
    if not enabled:
        return None
    if resolve_rerank_backend() == "dashscope":
        return DashScopeReranker()
    return CrossEncoderReranker()
