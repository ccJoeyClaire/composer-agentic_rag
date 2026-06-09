from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.metadata_schema import DEFAULT_MAX_RAG_ATTEMPTS, get_metadata
from agent.nodes import llm_node, tool_node
from agent.reflection.feedback import (
    FeedbackConfig,
    detect_feedback_node,
    plan_feedback_node,
    route_after_detect,
)
from agent.reflection.self_rag import (
    SelfRagConfig,
    self_rag_post_node,
    self_rag_pre_node,
)
from agent.state import AgentState
from agent.subgraph.CRAG import CragConfig, build_crag_node
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
        feedback_config = FeedbackConfig(llm=config.llm)
        graph.add_node(
            "detect_feedback",
            partial(
                detect_feedback_node,
                llm=feedback_config.llm,
                detect_fn=feedback_config.detect_fn,
            ),
        )
        graph.add_node(
            "plan_feedback",
            partial(
                plan_feedback_node,
                llm=feedback_config.llm,
                plan_fn=feedback_config.plan_fn,
            ),
        )

    if use_self_rag:
        self_rag_config = SelfRagConfig(
            llm=config.llm,
            max_rag_attempts=config.max_rag_attempts,
        )
        graph.add_node(
            "self_rag_pre",
            partial(
                self_rag_pre_node,
                llm=self_rag_config.llm,
                classify_fn=self_rag_config.classify_fn,
                max_rag_attempts=self_rag_config.max_rag_attempts,
            ),
        )
        graph.add_node(
            "self_rag_post",
            partial(
                self_rag_post_node,
                llm=self_rag_config.llm,
                grounded_fn=self_rag_config.grounded_fn,
                max_rag_attempts=self_rag_config.max_rag_attempts,
            ),
        )

    if use_crag:
        graph.add_node(
            "crag_eval",
            build_crag_node(
                CragConfig(
                    llm=config.llm,
                    max_rag_attempts=config.max_rag_attempts,
                    tool_box=tool_box,
                )
            ),
        )

    next_after_feedback = _next_node_after_feedback(use_self_rag=use_self_rag)
    if use_feedback:
        graph.set_entry_point("detect_feedback")
        graph.add_conditional_edges(
            "detect_feedback",
            route_after_detect,
            {"plan_feedback": "plan_feedback", "continue": next_after_feedback},
        )
        graph.add_edge("plan_feedback", next_after_feedback)
    elif use_self_rag:
        graph.set_entry_point("self_rag_pre")
    else:
        graph.set_entry_point("llm")

    if use_self_rag:
        graph.add_edge("self_rag_pre", "llm")

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
