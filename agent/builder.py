"""Compose the agent LangGraph from core nodes + optional capabilities."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.capabilities.human_feedback.capability import HumanFeedbackCapability
from agent.capabilities.rag_profile_router.capability import RagProfileRouterCapability
from agent.capabilities.retrieval_gate.capability import RetrievalGateCapability
from agent.config import AgentConfig
from agent.core.edges.after_llm import route_after_llm
from agent.core.edges.after_tools import route_after_tools
from agent.core.edges.names import NodeName
from agent.core.nodes.llm import llm_node
from agent.core.nodes.tools import tool_node
from agent.core.state import AgentState
from agent.core.tool_box import AgentToolBox
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
        enable_rag_profile_router=config.enable_rag_profile_router,
        rag_tool_name=config.rag_tool_name,
        extra_tools=extra_tools,
    )


def _resolve_checkpointer(config: AgentConfig) -> object | None:
    if config.checkpointer is not None:
        return config.checkpointer
    if config.enable_human_feedback:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    return None


def _register_capabilities(graph: StateGraph, config: AgentConfig) -> None:
    if config.enable_rag_profile_router:
        RagProfileRouterCapability().register(graph, config)
    if config.enable_retrieval_gate:
        RetrievalGateCapability().register(graph, config)


def _wire_edges(graph: StateGraph, config: AgentConfig) -> None:
    llm_route = partial(route_after_llm, agent_config=config)
    tools_route = partial(route_after_tools, agent_config=config)

    llm_targets: dict[str, str] = {
        NodeName.TOOLS: NodeName.TOOLS,
        END: END,
    }
    if config.enable_rag_profile_router:
        llm_targets[NodeName.RAG_PROFILE_ROUTER] = NodeName.RAG_PROFILE_ROUTER

    graph.add_conditional_edges(NodeName.LLM, llm_route, llm_targets)

    if config.enable_rag_profile_router:
        graph.add_edge(NodeName.RAG_PROFILE_ROUTER, NodeName.TOOLS)

    tools_targets: dict[str, str] = {NodeName.LLM: NodeName.LLM}
    if config.enable_retrieval_gate:
        tools_targets[NodeName.RETRIEVAL_GATE] = NodeName.RETRIEVAL_GATE

    graph.add_conditional_edges(NodeName.TOOLS, tools_route, tools_targets)

    if config.enable_retrieval_gate:
        graph.add_edge(NodeName.RETRIEVAL_GATE, NodeName.LLM)


def build_agent(config: AgentConfig) -> CompiledStateGraph:
    """Build and compile the agent graph."""
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

    return graph.compile(checkpointer=_resolve_checkpointer(config))
