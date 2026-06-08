"""CRAG subgraph — evaluate retrieval quality after RAG_search_tool (Phase 1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from typing import Awaitable, Callable

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

ScorePassagesFn = Callable[[str, list[str]], Awaitable[list[dict]]]

CRAG_SCORE_PROMPT = """You evaluate whether each retrieved passage helps answer the query.

Query:
{query}

Passages:
{passages}

Return JSON only:
{{"labels": [{{"index": 0, "label": "correct"|"incorrect"|"ambiguous"}}, ...]}}
"""


@dataclass
class CragConfig:
    llm: LLMClient | None = None
    rag_tool_name: str = DEFAULT_RAG_TOOL_NAME
    max_rag_attempts: int = DEFAULT_MAX_RAG_ATTEMPTS
    score_fn: ScorePassagesFn | None = None


def _normalize_labels(labels: list[dict], passage_count: int) -> list[dict]:
    by_index = {int(item.get("index", -1)): item for item in labels}
    normalized: list[dict] = []
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
    passages: list[str],
) -> list[dict]:
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


async def extract_rag_context(state: AgentState, *, rag_tool_name: str) -> dict:
    meta = dict(state.get("metadata") or {})
    hits = extract_rag_tool_results(state["messages"], rag_tool_name=rag_tool_name)
    if not hits:
        return merge_metadata(
            state,
            {
                "crag_verdict": "skipped",
                "crag_action": "use",
                "crag_labels": [],
            },
        )

    hit = hits[-1]
    rag_attempt = int(meta.get("rag_attempt", 0)) + 1
    return merge_metadata(
        state,
        {
            "rag_tool_name": rag_tool_name,
            "rag_attempt": rag_attempt,
            "rag_last_query": hit["query"],
            "rag_last_raw": hit["raw"],
        },
    )


def route_after_extract(state: AgentState) -> str:
    meta = state.get("metadata") or {}
    if meta.get("crag_verdict") == "skipped":
        return "crag_exit"
    return "score_relevance"


async def score_relevance(
    state: AgentState,
    *,
    llm: LLMClient | None,
    score_fn: ScorePassagesFn | None,
) -> dict:
    meta = state.get("metadata") or {}
    query = str(meta.get("rag_last_query", ""))
    raw = str(meta.get("rag_last_raw") or "")
    passages = split_rag_chunks(raw)

    if score_fn is not None:
        labels = _normalize_labels(await score_fn(query, passages), len(passages))
    elif llm is not None:
        labels = await default_score_passages(llm, query, passages)
    else:
        labels = [{"index": i, "label": "ambiguous"} for i in range(len(passages))]

    return merge_metadata(state, {"crag_labels": labels})


async def route_verdict(state: AgentState, *, max_rag_attempts: int) -> dict:
    meta = state.get("metadata") or {}
    labels = meta.get("crag_labels") or []
    rag_attempt = int(meta.get("rag_attempt", 0))

    counts = {"correct": 0, "incorrect": 0, "ambiguous": 0}
    for item in labels:
        label = str(item.get("label", "ambiguous"))
        counts[label] = counts.get(label, 0) + 1

    if counts["incorrect"] > 0:
        overall = "incorrect"
    elif counts["ambiguous"] > counts["correct"]:
        overall = "ambiguous"
    else:
        overall = "correct"

    if overall == "correct":
        action = "use"
    elif rag_attempt >= max_rag_attempts:
        action = "degrade"
    elif overall == "incorrect":
        action = "requery"
    else:
        action = "requery"

    patch: dict = {"crag_verdict": overall, "crag_action": action}
    if action == "requery":
        query = str(meta.get("rag_last_query", ""))
        patch["crag_requery_hint"] = (
            f"Previous retrieval may be insufficient. Try a more specific search for: {query}"
        )
    return merge_metadata(state, patch)


def route_after_verdict(state: AgentState) -> str:
    meta = state.get("metadata") or {}
    action = meta.get("crag_action", "use")
    if action == "use":
        return "trim_context"
    return "crag_exit"


async def trim_context(state: AgentState, *, rag_tool_name: str) -> dict:
    meta = state.get("metadata") or {}
    raw = str(meta.get("rag_last_raw") or "")
    passages = split_rag_chunks(raw)
    labels = meta.get("crag_labels") or []

    correct_indices = {
        int(item["index"])
        for item in labels
        if str(item.get("label")) == "correct"
    }
    if correct_indices:
        kept = [passages[i] for i in sorted(correct_indices) if i < len(passages)]
    elif meta.get("crag_action") == "degrade":
        kept = []
    else:
        kept = passages

    trimmed = "\n\n---\n\n".join(kept)
    updates: dict = merge_metadata(state, {"rag_last_raw": trimmed})

    hits = extract_rag_tool_results(state["messages"], rag_tool_name=rag_tool_name)
    if not hits:
        return updates

    tool_call_id = hits[-1]["tool_call_id"]
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_call_id != tool_call_id:
            continue
        new_tool_message = ToolMessage(
            content=trimmed or "No sufficiently relevant context found.",
            tool_call_id=tool_call_id,
            id=message.id,
        )
        if message.id:
            updates["messages"] = [RemoveMessage(id=message.id), new_tool_message]
        break
    return updates


async def crag_exit(state: AgentState) -> dict:
    return {}


def build_crag_subgraph(config: CragConfig):
    graph = StateGraph(AgentState)

    graph.add_node(
        "extract_rag_context",
        partial(extract_rag_context, rag_tool_name=config.rag_tool_name),
    )
    graph.add_node(
        "score_relevance",
        partial(
            score_relevance,
            llm=config.llm,
            score_fn=config.score_fn,
        ),
    )
    graph.add_node(
        "route_verdict",
        partial(route_verdict, max_rag_attempts=config.max_rag_attempts),
    )
    graph.add_node(
        "trim_context",
        partial(trim_context, rag_tool_name=config.rag_tool_name),
    )
    graph.add_node("crag_exit", crag_exit)

    graph.set_entry_point("extract_rag_context")
    graph.add_conditional_edges(
        "extract_rag_context",
        route_after_extract,
        {"crag_exit": "crag_exit", "score_relevance": "score_relevance"},
    )
    graph.add_edge("score_relevance", "route_verdict")
    graph.add_conditional_edges(
        "route_verdict",
        route_after_verdict,
        {"trim_context": "trim_context", "crag_exit": "crag_exit"},
    )
    graph.add_edge("trim_context", "crag_exit")
    graph.add_edge("crag_exit", END)

    return graph.compile()
