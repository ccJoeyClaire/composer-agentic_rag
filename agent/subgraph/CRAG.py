"""CRAG — isolated reflection subgraph with a private state.

The subgraph evaluates retrieval quality and, when passages score poorly, runs an
internal re-retrieval loop (escalating query-time variants via ``RAG_search_tool``,
then an optional reserved web fallback). All intermediate state lives in the private
:class:`CragState`; only the final qualified context is written back to ``AgentState``
by the parent wrapper node (:func:`build_crag_node`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Awaitable, Callable, List, Optional, Tuple, TypedDict

from langchain_core.messages import RemoveMessage, ToolMessage
from langgraph.graph import END, StateGraph

from agent.metadata_schema import (
    DEFAULT_MAX_RAG_ATTEMPTS,
    DEFAULT_RAG_TOOL_NAME,
    merge_metadata,
)
from agent.reflection.parsers import extract_rag_tool_results, split_rag_chunks
from agent.state import AgentState
from llm.client import LLMClient

ScorePassagesFn = Callable[[str, List[str]], Awaitable[List[dict]]]
RetrieveFn = Callable[[str, dict], Awaitable[str]]
WebFn = Callable[[str], Awaitable[str]]

# Query-time escalation ladder: each step is RAG_search_tool kwargs (reuses WS1).
DEFAULT_ESCALATION: Tuple[dict, ...] = (
    {"use_reranker": True, "recall_n": 50},
    {"use_hyde": True, "use_reranker": True, "recall_n": 50},
)

CRAG_SCORE_PROMPT = """You evaluate whether each retrieved passage helps answer the query.

Query:
{query}

Passages:
{passages}

Return JSON only:
{{"labels": [{{"index": 0, "label": "correct"|"incorrect"|"ambiguous"}}, ...]}}
"""


class CragState(TypedDict, total=False):
    """Private CRAG process state (NOT the parent AgentState)."""

    query: str
    passages: List[str]
    labels: List[dict]
    attempt: int
    max_attempts: int
    methods_tried: List[str]
    web_used: bool
    verdict: str
    action: str
    final_context: str


@dataclass
class CragConfig:
    llm: LLMClient | None = None
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME
    max_rag_attempts: int = DEFAULT_MAX_RAG_ATTEMPTS
    score_fn: ScorePassagesFn | None = None
    tool_box: object | None = None
    retrieve_fn: RetrieveFn | None = None
    web_enabled: bool = False
    web_tool_name: str = "bocha"
    web_fn: WebFn | None = None
    escalation: Tuple[dict, ...] = field(default_factory=lambda: DEFAULT_ESCALATION)


def _normalize_labels(labels: List[dict], passage_count: int) -> List[dict]:
    by_index = {int(item.get("index", -1)): item for item in labels}
    normalized: List[dict] = []
    for index in range(passage_count):
        item = by_index.get(index, {})
        label = str(item.get("label", "ambiguous")).lower()
        if label not in {"correct", "incorrect", "ambiguous"}:
            label = "ambiguous"
        normalized.append({"index": index, "label": label})
    return normalized


async def default_score_passages(
    llm: LLMClient,
    query: str,
    passages: List[str],
) -> List[dict]:
    if not passages:
        return []

    numbered = "\n".join(f"[{i}] {text[:800]}" for i, text in enumerate(passages))
    response = await llm.arequest_llm(
        [
            {
                "role": "user",
                "content": CRAG_SCORE_PROMPT.format(query=query, passages=numbered),
            }
        ],
        json_output=True,
    )
    try:
        payload = json.loads(response.content or "{}")
    except json.JSONDecodeError:
        payload = {}
    return _normalize_labels(payload.get("labels", []), len(passages))


def compute_verdict(labels: List[dict]) -> str:
    counts = {"correct": 0, "incorrect": 0, "ambiguous": 0}
    for item in labels:
        label = str(item.get("label", "ambiguous"))
        counts[label] = counts.get(label, 0) + 1

    if counts["incorrect"] > 0:
        return "incorrect"
    if counts["ambiguous"] > counts["correct"]:
        return "ambiguous"
    return "correct"


def decide_action(
    verdict: str,
    *,
    attempt: int,
    max_attempts: int,
    web_enabled: bool,
    web_used: bool,
) -> str:
    if verdict == "correct":
        return "use"
    if attempt < max_attempts:
        return "requery"
    if web_enabled and not web_used:
        return "web_fallback"
    return "degrade"


async def _do_retrieve(config: CragConfig, query: str, options: dict) -> str:
    if config.retrieve_fn is not None:
        return await config.retrieve_fn(query, options)
    if config.tool_box is not None:
        result = await config.tool_box.ainvoke(
            config.rag_tool_name, {"query": query, **options}
        )
        if getattr(result, "error", None):
            return ""
        return str(getattr(result, "output", "") or "")
    return ""


async def _do_web(config: CragConfig, query: str) -> str:
    if config.web_fn is not None:
        return await config.web_fn(query)
    if config.tool_box is not None:
        result = await config.tool_box.ainvoke(config.web_tool_name, {"query": query})
        if getattr(result, "error", None):
            return ""
        return str(getattr(result, "output", "") or "")
    return ""


# -- subgraph nodes (operate on CragState) ----------------------------------


async def score_node(
    state: CragState,
    *,
    llm: LLMClient | None,
    score_fn: ScorePassagesFn | None,
) -> dict:
    query = state.get("query", "")
    passages = state.get("passages", [])

    if score_fn is not None:
        labels = _normalize_labels(await score_fn(query, passages), len(passages))
    elif llm is not None:
        labels = await default_score_passages(llm, query, passages)
    else:
        labels = [{"index": i, "label": "ambiguous"} for i in range(len(passages))]

    return {"labels": labels}


async def verdict_node(state: CragState, *, web_enabled: bool) -> dict:
    verdict = compute_verdict(state.get("labels", []))
    action = decide_action(
        verdict,
        attempt=int(state.get("attempt", 0)),
        max_attempts=int(state.get("max_attempts", DEFAULT_MAX_RAG_ATTEMPTS)),
        web_enabled=web_enabled,
        web_used=bool(state.get("web_used", False)),
    )
    return {"verdict": verdict, "action": action}


def route_after_verdict(state: CragState) -> str:
    action = state.get("action", "use")
    if action == "requery":
        return "reretrieve"
    if action == "web_fallback":
        return "web"
    return "finalize"


async def reretrieve_node(state: CragState, *, crag_config: CragConfig) -> dict:
    attempt = int(state.get("attempt", 1))
    ladder = crag_config.escalation or DEFAULT_ESCALATION
    options = dict(ladder[min(attempt - 1, len(ladder) - 1)])
    query = state.get("query", "")

    raw = await _do_retrieve(crag_config, query, options)
    passages = split_rag_chunks(raw)

    methods = list(state.get("methods_tried", []))
    methods.append(",".join(f"{k}={v}" for k, v in sorted(options.items())))

    return {
        "passages": passages if passages else state.get("passages", []),
        "attempt": attempt + 1,
        "methods_tried": methods,
    }


async def web_node(state: CragState, *, crag_config: CragConfig) -> dict:
    query = state.get("query", "")
    raw = await _do_web(crag_config, query)
    passages = split_rag_chunks(raw)
    methods = list(state.get("methods_tried", []))
    methods.append("web")
    return {
        "passages": passages if passages else state.get("passages", []),
        "web_used": True,
        "methods_tried": methods,
    }


async def finalize_node(state: CragState) -> dict:
    passages = state.get("passages", [])
    labels = state.get("labels", [])
    action = state.get("action", "use")

    correct_indices = {
        int(item["index"])
        for item in labels
        if str(item.get("label")) == "correct"
    }
    if state.get("web_used"):
        kept = passages
    elif correct_indices:
        kept = [passages[i] for i in sorted(correct_indices) if i < len(passages)]
    elif action == "degrade":
        kept = []
    else:
        kept = passages

    return {"final_context": "\n\n---\n\n".join(kept)}


def build_crag_subgraph(config: CragConfig):
    graph = StateGraph(CragState)

    graph.add_node(
        "score",
        partial(score_node, llm=config.llm, score_fn=config.score_fn),
    )
    graph.add_node("verdict", partial(verdict_node, web_enabled=config.web_enabled))
    graph.add_node("reretrieve", partial(reretrieve_node, crag_config=config))
    graph.add_node("web", partial(web_node, crag_config=config))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("score")
    graph.add_edge("score", "verdict")
    graph.add_conditional_edges(
        "verdict",
        route_after_verdict,
        {"reretrieve": "reretrieve", "web": "web", "finalize": "finalize"},
    )
    graph.add_edge("reretrieve", "score")
    graph.add_edge("web", "score")
    graph.add_edge("finalize", END)

    return graph.compile()


def _rewrite_tool_message(
    state: AgentState,
    *,
    tool_call_id: str,
    content: str,
) -> Optional[list]:
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_call_id != tool_call_id:
            continue
        new_tool_message = ToolMessage(
            content=content or "No sufficiently relevant context found.",
            tool_call_id=tool_call_id,
            id=message.id,
        )
        if message.id:
            return [RemoveMessage(id=message.id), new_tool_message]
        break
    return None


def build_crag_node(config: CragConfig):
    """Parent wrapper: map AgentState -> CragState, run subgraph, write back only the
    final qualified context (rewrite the latest RAG ToolMessage + minimal metadata)."""
    subgraph = build_crag_subgraph(config)

    async def crag_node(state: AgentState) -> dict:
        hits = extract_rag_tool_results(state["messages"], rag_tool_name=config.rag_tool_name)
        if not hits:
            return merge_metadata(
                state,
                {"crag_verdict": "skipped", "crag_action": "use", "crag_labels": []},
            )

        hit = hits[-1]
        meta = state.get("metadata") or {}
        attempt = int(meta.get("rag_attempt", 0)) + 1

        init: CragState = {
            "query": hit["query"],
            "passages": split_rag_chunks(hit["raw"]),
            "attempt": attempt,
            "max_attempts": config.max_rag_attempts,
            "methods_tried": [],
            "web_used": False,
        }
        final = await subgraph.ainvoke(init)

        final_context = final.get("final_context", "")
        updates = merge_metadata(
            state,
            {
                "rag_tool_name": config.rag_tool_name,
                "rag_attempt": int(final.get("attempt", attempt)),
                "rag_last_query": hit["query"],
                "rag_last_raw": final_context,
                "crag_verdict": final.get("verdict", "ambiguous"),
                "crag_action": final.get("action", "use"),
            },
        )

        rewritten = _rewrite_tool_message(
            state,
            tool_call_id=hit["tool_call_id"],
            content=final_context,
        )
        if rewritten is not None:
            updates["messages"] = rewritten
        return updates

    return crag_node
