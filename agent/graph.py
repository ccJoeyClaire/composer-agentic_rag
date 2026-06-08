from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.metadata_schema import DEFAULT_MAX_RAG_ATTEMPTS, get_metadata
from agent.nodes import llm_node, tool_node
from agent.state import AgentState
from agent.subgraph.CRAG import CragConfig, build_crag_subgraph
from agent.subgraph.RAG_FeedBack import FeedbackConfig, build_feedback_subgraph
from agent.subgraph.Self_RAG import SelfRagConfig, build_self_rag_post_subgraph, build_self_rag_pre_subgraph
from llm.client import LLMClient
from tools.tool_box import ToolBox

AgentPattern = Literal[
    "react",
    "react_crag",
    "react_self_rag",
    "react_feedback",
    "react_full",
    "react_all",
]
SUPPORTED_PATTERNS: tuple[AgentPattern, ...] = (
    "react",
    "react_crag",
    "react_self_rag",
    "react_feedback",
    "react_full",
    "react_all",
)

ReActAgent = CompiledStateGraph


@dataclass
class AgentConfig:
    llm: LLMClient
    tool_box: ToolBox | None = None
    tool_calls: bool = True
    checkpointer: object | None = None
    enable_crag: bool = False
    enable_self_rag: bool = False
    enable_feedback: bool = False
    max_rag_attempts: int = DEFAULT_MAX_RAG_ATTEMPTS


def _pattern_flags(pattern: AgentPattern) -> tuple[bool, bool, bool]:
    if pattern == "react_crag":
        return True, False, False
    if pattern == "react_self_rag":
        return False, True, False
    if pattern == "react_feedback":
        return False, False, True
    if pattern == "react_full":
        return True, True, False
    if pattern == "react_all":
        return True, True, True
    return False, False, False


def _next_node_after_feedback(*, use_self_rag: bool) -> str:
    return "self_rag_pre" if use_self_rag else "llm"


def if_tool_calls(state: AgentState) -> str:
    """Route: if last message has tool_calls -> 'tools', else -> END."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def if_after_llm(state: AgentState, *, use_self_rag: bool) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    if use_self_rag:
        return "self_rag_post"
    return END


def route_after_crag(state: AgentState) -> str:
    """After CRAG, always return to llm (mode A requery)."""
    return "llm"


def route_after_self_rag_post(state: AgentState) -> str:
    """Retry via llm when answer is not grounded and attempts remain."""
    meta = get_metadata(state)
    if meta.get("self_rag_grounded") is False and meta.get("self_rag_retry_allowed", True):
        rag_attempt = int(meta.get("rag_attempt", 0))
        max_attempts = int(meta.get("max_rag_attempts", DEFAULT_MAX_RAG_ATTEMPTS))
        if rag_attempt < max_attempts:
            return "llm"
    return END


def build_ReAct_agent(
    config: AgentConfig,
    *,
    enable_crag: bool | None = None,
    enable_self_rag: bool | None = None,
    enable_feedback: bool | None = None,
) -> CompiledStateGraph:
    use_crag = config.enable_crag if enable_crag is None else enable_crag
    use_self_rag = config.enable_self_rag if enable_self_rag is None else enable_self_rag
    use_feedback = config.enable_feedback if enable_feedback is None else enable_feedback
    tool_box = config.tool_box or ToolBox()
    graph = StateGraph(AgentState)

    graph.add_node(
        "llm",
        partial(
            llm_node,
            llm=config.llm,
            tool_box=tool_box,
            tool_calls=config.tool_calls,
        ),
    )
    graph.add_node(
        "tools",
        partial(tool_node, tool_box=tool_box),
    )

    if use_feedback:
        graph.add_node(
            "feedback",
            build_feedback_subgraph(FeedbackConfig(llm=config.llm)),
        )

    if use_self_rag:
        self_rag_config = SelfRagConfig(
            llm=config.llm,
            max_rag_attempts=config.max_rag_attempts,
        )
        graph.add_node("self_rag_pre", build_self_rag_pre_subgraph(self_rag_config))
        graph.add_node("self_rag_post", build_self_rag_post_subgraph(self_rag_config))

    if use_crag:
        graph.add_node(
            "crag_eval",
            build_crag_subgraph(
                CragConfig(
                    llm=config.llm,
                    max_rag_attempts=config.max_rag_attempts,
                )
            ),
        )

    if use_feedback:
        entry = "feedback"
        graph.set_entry_point(entry)
        graph.add_edge("feedback", _next_node_after_feedback(use_self_rag=use_self_rag))
    elif use_self_rag:
        graph.set_entry_point("self_rag_pre")
        graph.add_edge("self_rag_pre", "llm")
    else:
        graph.set_entry_point("llm")

    after_llm = partial(if_after_llm, use_self_rag=use_self_rag)
    llm_targets: dict = {"tools": "tools", END: END}
    if use_self_rag:
        llm_targets["self_rag_post"] = "self_rag_post"
    graph.add_conditional_edges("llm", after_llm, llm_targets)

    if use_crag:
        graph.add_edge("tools", "crag_eval")
        graph.add_conditional_edges("crag_eval", route_after_crag, {"llm": "llm"})
    else:
        graph.add_edge("tools", "llm")

    if use_self_rag:
        graph.add_conditional_edges(
            "self_rag_post",
            route_after_self_rag_post,
            {"llm": "llm", END: END},
        )

    return graph.compile(checkpointer=config.checkpointer)


def build_agent(
    config: AgentConfig,
    *,
    pattern: AgentPattern = "react",
) -> CompiledStateGraph:
    if pattern not in SUPPORTED_PATTERNS:
        supported = ", ".join(SUPPORTED_PATTERNS)
        raise ValueError(f"Unknown agent pattern {pattern!r}. Supported: {supported}")

    use_crag, use_self_rag, use_feedback = _pattern_flags(pattern)
    use_crag = use_crag or config.enable_crag
    use_self_rag = use_self_rag or config.enable_self_rag
    use_feedback = use_feedback or config.enable_feedback
    return build_ReAct_agent(
        config,
        enable_crag=use_crag,
        enable_self_rag=use_self_rag,
        enable_feedback=use_feedback,
    )
