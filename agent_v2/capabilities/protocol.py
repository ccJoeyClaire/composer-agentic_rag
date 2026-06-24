"""Capability protocol — register optional nodes on the shared ReAct skeleton."""

from __future__ import annotations

from typing import Protocol

from langgraph.graph import StateGraph

from agent_v2.config import AgentConfig


class Capability(Protocol):
    """Graph node capability — registers a node; edges wired in ``builder``."""

    def register(self, graph: StateGraph, config: AgentConfig) -> None:
        """Add capability nodes to *graph* (edges wired centrally in builder)."""
        ...
