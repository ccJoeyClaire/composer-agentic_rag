"""Agent graph state — the single place to read before touching the graph.

Contains ``AgentState``, reflection metadata schema, merge helpers, and
message construction templates.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict, cast

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, List

# ---------------------------------------------------------------------------
# Reflection metadata (CRAG / Self-RAG / Feedback)
# ---------------------------------------------------------------------------

CragVerdict = Literal["correct", "incorrect", "ambiguous", "skipped"]
CragAction = Literal["use", "requery", "web_fallback", "degrade"]
FeedbackKind = Literal["correction", "thumbs_down", "clarify"]
FeedbackAction = Literal["none", "requery", "rewrite", "clarify"]


class CragLabel(TypedDict):
    index: int
    label: CragVerdict
    score: NotRequired[float]


class AgentMetadata(TypedDict, total=False):
    # RAG invocation
    rag_tool_name: str
    rag_attempt: int
    rag_last_query: str
    rag_last_raw: str | None
    max_rag_attempts: int

    # CRAG (#17)
    crag_verdict: CragVerdict
    crag_labels: list[CragLabel]
    crag_action: CragAction
    crag_requery_hint: str | None

    # Self-RAG (#12)
    self_rag_need_retrieve: bool | None
    self_rag_grounded: bool | None
    self_rag_retry_allowed: bool
    self_rag_retrieve_hint: str | None
    self_rag_retry_hint: str | None

    # Feedback (#11)
    feedback_detected: bool
    feedback_kind: FeedbackKind | None
    feedback_action: FeedbackAction
    feedback_suggested_query: str | None
    feedback_hint: str | None


DEFAULT_RAG_TOOL_NAME = "RAG_search_tool"
DEFAULT_WEB_TOOL_NAME = "tavily_search"
DEFAULT_MAX_RAG_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    # Conversation history. Concrete elements are always one of
    # Human/AI/System/ToolMessage; ``add_messages`` is the reducer that appends
    # whatever a node returns under ``{"messages": [...]}`` instead of overwriting.
    messages: Annotated[List[BaseMessage], add_messages]
    # Reflection metadata accumulated across pipeline stages. Every possible key
    # is documented in AgentMetadata above.
    metadata: NotRequired[AgentMetadata]
    error: NotRequired[str]


def get_metadata(state: AgentState) -> AgentMetadata:
    """Return a shallow copy of the state's metadata so callers can mutate freely.

    Returns an empty ``AgentMetadata`` when the state has no metadata yet.
    """
    return cast("AgentMetadata", dict(state.get("metadata") or {}))


def merge_metadata(state: AgentState, patch: AgentMetadata) -> dict[str, AgentMetadata]:
    """Merge ``patch`` into the current metadata and return a state update.

    Args:
        state: Current agent state (read-only here).
        patch: Partial metadata fields to overwrite/add.

    Returns:
        A ``{"metadata": ...}`` dict suitable for returning from a graph node.
    """
    current = get_metadata(state)
    current.update(patch)
    return {"metadata": current}


# ---------------------------------------------------------------------------
# Message templates (reference only — run ``python -m agent.state`` to inspect)
#
# All four types inherit BaseMessage fields:
#   content: str | list[str | dict]
#   type: "human" | "system" | "ai" | "tool"
#   name: str | None          (optional)
#   id: str | None            (optional)
#
# BaseMessage helpers (built into LangChain — every message has these):
#   msg.text            — plain text (property; handles block-list content)
#   msg.pretty_repr()   — debug string with titled header
#   msg.pretty_print()  — print pretty_repr to stdout
#
# Typical ReAct turn:
#   HumanMessage → AIMessage(tool_calls=[...]) → ToolMessage → AIMessage(answer)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    # 1. SystemMessage — system prompt (usually first in the list)
    _system = SystemMessage(content="You are a helpful RAG assistant.")

    # 2. HumanMessage — user input
    _human = HumanMessage(content="What is the capital of France?")

    # 3. AIMessage — model reply; tool_calls present when the model wants to call tools
    _ai_tool = AIMessage(
        content="",
        tool_calls=[
            {
                "name": DEFAULT_RAG_TOOL_NAME,
                "args": {"query": "capital of France"},
                "id": "call_abc123",
            }
        ],
    )

    # 4. ToolMessage — tool result; tool_call_id MUST match AIMessage.tool_calls[i]["id"]
    _tool = ToolMessage(
        content="Paris is the capital of France.",
        tool_call_id="call_abc123",
    )

    # Final answer (no tool_calls → graph routes to END)
    _ai_answer = AIMessage(content="The capital of France is Paris.")

    _MESSAGE_SHAPE_REF = (_system, _human, _ai_tool, _tool, _ai_answer)

    # BaseMessage.pretty_repr() — returns a formatted debug string
    print(_human.pretty_repr())

    # BaseMessage.pretty_print() — prints pretty_repr (AIMessage also shows tool_calls)
    _ai_tool.pretty_print()

    # msg.text — extract plain text without printing
    print("text:", _human.text)

    # Print the full ReAct turn
    for _msg in _MESSAGE_SHAPE_REF:
        _msg.pretty_print()

