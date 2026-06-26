"""Shared machinery for reflection-pattern agent eval on pooled BEIR data."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from legacy.agent.graph import AgentConfig, build_agent
from legacy.agent.state import DEFAULT_RAG_TOOL_NAME
from llm.client import LLMClient
from rag.base import Chunk
from rag.core import RAGIndexer, RAGRetriever
from tools.LocalTool.RAG_tool import bind_retriever
from tools.tool_box import ToolBox

from _eval_.agent_eval.judge import judge_answer
from _eval_.config import AGENT_SYSTEM_PROMPT, AgentRunConfig
from _eval_.data_preparing.beir import (
    CorpusDoc,
    DocId,
    EvalQuery,
    Qrels,
    QueryId,
    resolve_chunk_doc_id,
)
from _eval_.data_preparing.pooling import gold_docs
from _eval_.scoring.metrics import mean_metrics, query_metrics

_LOCAL_TOOL_PACKAGES = ("tools.LocalTool",)
BASELINE_PATTERN = "react"


async def drop_collection(store) -> None:
    """Delete the backing collection when it exists."""
    if await store.client.collection_exists(store.collection):
        await store.client.delete_collection(store.collection)


async def collection_exists(store) -> bool:
    """Return whether the store's backing collection is already present."""
    return await store.client.collection_exists(store.collection)


async def index_doc_list(
    indexer: RAGIndexer,
    docs: list[CorpusDoc],
    *,
    concurrency: int,
) -> int:
    """Index a list of docs with bounded concurrency; returns docs indexed."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _index_one(doc: CorpusDoc) -> None:
        async with semaphore:
            source = doc.title or doc.doc_id
            await indexer.aindex(doc.text, source=source, doc_id=doc.doc_id)

    await asyncio.gather(*(_index_one(doc) for doc in docs))
    return len(docs)


class AgentQueryScore(TypedDict):
    query_id: str
    num_gold: int
    answer: str
    metrics: dict[str, float]
    signals: dict[str, object]


@dataclass(frozen=True)
class AgentPatternResult:
    """Aggregate outcome for one agent pattern over the evaluated queries."""

    pattern: str
    rag_profile: str
    num_queries: int
    mean_metrics: dict[str, float]
    per_query: list[AgentQueryScore]


class RecordingRetriever:
    """Wraps a ``RAGRetriever`` and records the doc ids of every retrieval."""

    def __init__(self, inner: RAGRetriever) -> None:
        self.inner = inner
        self._calls: list[list[DocId]] = []

    def reset(self) -> None:
        self._calls = []

    async def aquery(self, query: str, top_k: int | None = None) -> list[Chunk]:
        chunks = await self.inner.aquery(query, top_k=top_k)
        self._calls.append(
            [
                doc_id
                for chunk in chunks
                if (doc_id := resolve_chunk_doc_id(chunk.metadata))
            ]
        )
        return chunks

    def retrieved_doc_ids(self) -> list[DocId]:
        """Union of retrieved doc ids across all calls, first-seen order."""
        seen: set[DocId] = set()
        ranked: list[DocId] = []
        for call in self._calls:
            for doc_id in call:
                if doc_id not in seen:
                    seen.add(doc_id)
                    ranked.append(doc_id)
        return ranked

    @property
    def num_calls(self) -> int:
        return len(self._calls)


def final_answer(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and (message.content or "").strip():
            return str(message.content)
    return ""


def process_signals(messages: list[BaseMessage], metadata: dict) -> dict[str, object]:
    """Count turns/tool calls and surface reflection verdicts from metadata."""
    num_turns = 0
    num_tool_calls = 0
    num_rag_calls = 0
    for message in messages:
        if isinstance(message, AIMessage):
            num_turns += 1
            for call in message.tool_calls or []:
                num_tool_calls += 1
                if call.get("name") == DEFAULT_RAG_TOOL_NAME:
                    num_rag_calls += 1
    return {
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "num_rag_calls": num_rag_calls,
        "rag_attempt": metadata.get("rag_attempt"),
        "crag_verdict": metadata.get("crag_verdict"),
        "crag_action": metadata.get("crag_action"),
        "self_rag_grounded": metadata.get("self_rag_grounded"),
        "self_rag_need_retrieve": metadata.get("self_rag_need_retrieve"),
        "feedback_action": metadata.get("feedback_action"),
    }


async def score_one_query(
    graph,
    recorder: RecordingRetriever,
    query: EvalQuery,
    relevance: dict[DocId, int],
    *,
    cfg: AgentRunConfig,
    llm: LLMClient,
    gold_texts: list[str],
) -> AgentQueryScore:
    recorder.reset()
    state = await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=query.text),
            ]
        },
        config={"recursion_limit": cfg.recursion_limit},
    )
    messages: list[BaseMessage] = state["messages"]
    metadata = dict(state.get("metadata") or {})

    answer = final_answer(messages)
    gold = gold_docs(relevance, cfg.pool_spec.rel_threshold)
    ranked = recorder.retrieved_doc_ids()

    metrics: dict[str, float] = {
        f"ctx_{name}": value
        for name, value in query_metrics(
            ranked, relevance, gold, k_values=cfg.k_values, mrr_k=cfg.max_k
        ).items()
    }
    signals = process_signals(messages, metadata)
    metrics["ctx_docs"] = float(len(ranked))
    metrics["num_rag_calls"] = float(signals["num_rag_calls"])
    metrics["num_tool_calls"] = float(signals["num_tool_calls"])
    metrics["num_turns"] = float(signals["num_turns"])
    metrics["answer_len"] = float(len(answer))
    metrics["has_answer"] = 1.0 if answer.strip() else 0.0

    if cfg.use_judge:
        verdict = await judge_answer(
            llm, query=query.text, answer=answer, gold_texts=gold_texts
        )
        metrics["judge_correct"] = 1.0 if verdict["correct"] else 0.0
        metrics["judge_grounded"] = 1.0 if verdict["grounded"] else 0.0
        signals["judge_reason"] = verdict["reason"]

    return AgentQueryScore(
        query_id=query.query_id,
        num_gold=len(gold),
        answer=answer,
        metrics=metrics,
        signals=signals,
    )


async def evaluate_pattern(
    pattern: str,
    *,
    retriever: RAGRetriever,
    queries: dict[QueryId, EvalQuery],
    qrels: Qrels,
    query_ids: list[QueryId],
    gold_texts: dict[QueryId, list[str]],
    cfg: AgentRunConfig,
) -> AgentPatternResult:
    """Run ``pattern`` over every evaluated query against a shared retriever."""
    recorder = RecordingRetriever(retriever)
    bind_retriever(recorder, top_k=cfg.agent_rag_top_k)  # type: ignore[arg-type]

    llm = LLMClient()
    tool_box = ToolBox(packages=_LOCAL_TOOL_PACKAGES)
    graph = build_agent(AgentConfig(llm=llm, tool_box=tool_box), pattern=pattern)

    per_query: list[AgentQueryScore] = []
    for qid in query_ids:
        score = await score_one_query(
            graph,
            recorder,
            queries[qid],
            qrels[qid],
            cfg=cfg,
            llm=llm,
            gold_texts=gold_texts.get(qid, []),
        )
        per_query.append(score)

    return AgentPatternResult(
        pattern=pattern,
        rag_profile=cfg.rag_profile,
        num_queries=len(per_query),
        mean_metrics=mean_metrics([q["metrics"] for q in per_query]),
        per_query=per_query,
    )


def signal_rate(per_query: list[AgentQueryScore], key: str, value: object) -> float:
    """Fraction of queries where ``signals[key]`` equals ``value``."""
    if not per_query:
        return 0.0
    hits = sum(1 for row in per_query if row["signals"].get(key) == value)
    return hits / len(per_query)


def metric_deltas(
    pattern_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    keys: tuple[str, ...],
) -> dict[str, float]:
    """``pattern - baseline`` for each mean metric key."""
    return {f"delta_{key}": pattern_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0) for key in keys}
