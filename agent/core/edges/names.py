"""Canonical node names for the agent graph."""

from __future__ import annotations


class NodeName:
    """String constants for LangGraph node registration and routing."""

    LLM = "llm"
    TOOLS = "tools"
    RAG_PROFILE_ROUTER = "rag_profile_router"
    RETRIEVAL_GATE = "retrieval_gate"
