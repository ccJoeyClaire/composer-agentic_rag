"""Compose the agent_v2 LangGraph from core nodes + optional capabilities."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_v2.capabilities.human_feedback.capability import HumanFeedbackCapability
from agent_v2.capabilities.rag_profile.capability import RagProfileCapability
from agent_v2.capabilities.retrieval_gate.capability import RetrievalGateCapability
from agent_v2.config import AgentConfig
from agent_v2.core.edges.after_llm import route_after_llm
from agent_v2.core.edges.after_tools import route_after_tools
from agent_v2.core.edges.names import NodeName
from agent_v2.core.nodes.llm import llm_node
from agent_v2.core.nodes.tools import tool_node
from agent_v2.core.state import AgentState
from agent_v2.core.tool_box import AgentToolBox
from tools.tool_box import ToolBox

ReActAgent = CompiledStateGraph


def _resolve_tool_box(config: AgentConfig) -> AgentToolBox:
    inner = config.tool_box or ToolBox()
    extra_tools: dict[str, object] = {}
    if config.enable_human_feedback:
        extra_tools.update(HumanFeedbackCapability.extra_tools())
    return AgentToolBox(
        inner=inner,
        enable_web_search=config.enable_web_search,
        web_tool_name=config.web_tool_name,
        extra_tools=extra_tools,
    )


def _register_capabilities(graph: StateGraph, config: AgentConfig) -> None:
    if config.enable_rag_profile:
        RagProfileCapability().register(graph, config)
    if config.enable_retrieval_gate:
        RetrievalGateCapability().register(graph, config)
    if config.enable_human_feedback:
        HumanFeedbackCapability().register(graph, config)


def _wire_edges(graph: StateGraph, config: AgentConfig) -> None:
    """Wire the LLM-centric topology (see agent_v2/README structure)."""
    llm_route = partial(route_after_llm, agent_config=config)
    tools_route = partial(route_after_tools, agent_config=config)

    llm_targets: dict[str, str] = {
        NodeName.TOOLS: NodeName.TOOLS,
        END: END,
    }
    if config.enable_rag_profile:
        llm_targets[NodeName.RAG_PROFILE_ROUTER] = NodeName.RAG_PROFILE_ROUTER

    graph.add_conditional_edges(NodeName.LLM, llm_route, llm_targets)

    if config.enable_rag_profile:
        graph.add_edge(NodeName.RAG_PROFILE_ROUTER, NodeName.TOOLS)

    tools_targets: dict[str, str] = {NodeName.LLM: NodeName.LLM}
    if config.enable_retrieval_gate:
        tools_targets[NodeName.RETRIEVAL_GATE] = NodeName.RETRIEVAL_GATE
    if config.enable_human_feedback:
        tools_targets[NodeName.HUMAN_FEEDBACK] = NodeName.HUMAN_FEEDBACK

    graph.add_conditional_edges(NodeName.TOOLS, tools_route, tools_targets)

    if config.enable_retrieval_gate:
        graph.add_edge(NodeName.RETRIEVAL_GATE, NodeName.LLM)

    if config.enable_human_feedback:
        graph.add_edge(NodeName.HUMAN_FEEDBACK, END)


def build_agent(config: AgentConfig) -> CompiledStateGraph:
    """Build and compile the agent_v2 graph.

    Topology (LLM as hub):
        entry → llm
        llm → rag_profile_router → tools   (RAG tool calls, profile enabled)
        llm → tools                        (other tool calls)
        llm → END                          (no tool calls — LLM decides done)
        tools → retrieval_gate → llm       (RAG batch, gate enabled)
        tools → human_feedback → END       (clarification tool, feedback enabled)
        tools → llm                        (web / other)
    """
    tool_box = _resolve_tool_box(config)
    graph = StateGraph(AgentState)

    graph.add_node(
        NodeName.LLM,
        partial(
            llm_node,
            llm=config.llm,
            tool_box=tool_box,
            tool_calls=True,
        ),
    )
    graph.add_node(
        NodeName.TOOLS,
        partial(tool_node, tool_box=tool_box),
    )

    _register_capabilities(graph, config)
    graph.set_entry_point(NodeName.LLM)
    _wire_edges(graph, config)

    return graph.compile(checkpointer=config.checkpointer)
