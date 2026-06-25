"""Feedback pattern — ``reflection_patterns.feedback`` (interrupt/resume)."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from agent_v2.patterns.config import RequestConfig, build_graph_for_pattern

PATTERN_ID = "feedback"


def build_graph(request_config: RequestConfig) -> CompiledStateGraph:
    return build_graph_for_pattern(PATTERN_ID, request_config)
