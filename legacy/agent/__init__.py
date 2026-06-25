"""Agent graph, nodes, and state."""

__all__ = [
    "AgentConfig",
    "ReActAgent",
    "build_ReAct_agent",
    "build_agent",
]


def __getattr__(name: str):
    if name in __all__:
        from legacy.agent.graph import AgentConfig, ReActAgent, build_ReAct_agent, build_agent

        return {
            "AgentConfig": AgentConfig,
            "ReActAgent": ReActAgent,
            "build_ReAct_agent": build_ReAct_agent,
            "build_agent": build_agent,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
