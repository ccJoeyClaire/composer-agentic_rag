"""Human feedback capability configuration."""

from __future__ import annotations

from dataclasses import dataclass

from llm.client import LLMClient


@dataclass
class HumanFeedbackConfig:
    """Settings for LLM-initiated clarification (optional tool + post-processor)."""

    llm: LLMClient | None = None
