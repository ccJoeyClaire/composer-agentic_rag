"""agent — LLM-centric ReAct agent with optional capabilities."""

__all__ = [
    "AgentConfig",
    "ReActAgent",
    "build_agent",
    "eval_config",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from agent.builder import ReActAgent, build_agent
        from agent.config import AgentConfig, eval_config

        return {
            "AgentConfig": AgentConfig,
            "ReActAgent": ReActAgent,
            "build_agent": build_agent,
            "eval_config": eval_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
