from __future__ import annotations

from typing import Literal, TypedDict

from agent.output import AgentRunRecord
from rag.serialize import RetrieveTraceEntry


# ---------------------------------------------------------------------------
# Gold set (QA generator output)
# ---------------------------------------------------------------------------


class GoldSample(TypedDict):
    """One gold row from the QA generator.

    ``user_question`` simulates end-user phrasing (agent eval input).
    ``gold_query`` is retrieval-ready text (direct RAG eval input).
    """

    query_id: str
    user_question: str
    gold_query: str
    gt_answer: str


# ---------------------------------------------------------------------------
# Eval matrix — one arm per runner_id
# ---------------------------------------------------------------------------


class AgentArmConfig(TypedDict, total=False):
    """Agent graph settings for ``mode='agent'`` runners."""

    pattern_id: str
    profile_id: str
    enable_web_search: bool


class RagArmConfig(TypedDict, total=False):
    """RAG retriever settings for ``mode='rag'`` runners."""

    profile_id: str
    top_k: int


class EvalRunner(TypedDict):
    """One infer arm: selects collection + agent or RAG profile."""

    runner_id: str
    mode: Literal["agent", "rag"]
    collection: str
    agent_config: AgentArmConfig
    rag_config: RagArmConfig

# ---------------------------------------------------------------------------
# Infer stage artifacts (per query, before assembly)
# ---------------------------------------------------------------------------


class AgentInferArtifact(TypedDict):
    """Agent infer dump: ``OutputState.to_record_schema()`` plus eval join keys.

    Written to ``data/infer/{runner_id}/agent/{query_id}.json``.
    ``run`` holds messages, metadata (incl. ``retrieved_context``), and
    ``final_message`` — the JSON-safe form of post-``ainvoke`` state, not live
    :class:`agent.core.state.AgentState`.
    """

    query_id: str
    invoked_query: str  # ``GoldSample.user_question``
    run: AgentRunRecord


class RagInferArtifact(TypedDict):
    """RAG infer dump: retrieval trace + LLM answer for one gold row.

    Written to ``data/infer/{runner_id}/rag/{query_id}.json``.
    ``trace`` matches ``get_start.retrieve_example`` stage snapshots;
    ``generator_response`` comes from LLM given ``user_question`` + ``final`` chunks.
    """

    query_id: str
    user_question: str   # LLM prompt context
    gold_query: str      # ``trace.query`` — what retrieval ran on
    trace: RetrieveTraceEntry
    generator_response: str


# ---------------------------------------------------------------------------
# RAGChecker wire format
# ---------------------------------------------------------------------------


class CheckerContextChunk(TypedDict):
    """One chunk in ``retrieved_context`` passed to RAGChecker."""

    doc_id: str  # ``{source}|{heading_path}``
    text: str    # raw chunk body (no contextual header prefix)


class CheckerSample(TypedDict):
    """One ``results[]`` row in a ``CheckerInput`` batch (gold + infer, pre-score)."""

    query_id: str
    gold_question: str
    gt_answer: str
    generator_response: str
    retrieved_context: list[CheckerContextChunk]


class CheckerInput(TypedDict):
    """Top-level JSON file for one ``runner_id`` (``evaluate()`` batch input)."""

    results: list[CheckerSample]


# ---------------------------------------------------------------------------
# RAGChecker metrics (score stage output)
# ---------------------------------------------------------------------------


class OverallMetrics(TypedDict, total=False):
    precision: float
    recall: float
    f1: float


class RetrieverMetrics(TypedDict, total=False):
    claim_recall: float
    context_precision: float


class GeneratorMetrics(TypedDict, total=False):
    context_utilization: float
    noise_sensitivity_in_relevant: float
    noise_sensitivity_in_irrelevant: float
    hallucination: float
    self_knowledge: float
    faithfulness: float


class CheckerRunMetrics(TypedDict, total=False):
    """Aggregate metrics from one ``evaluate()`` over a full ``CheckerInput``."""

    overall_metrics: OverallMetrics
    retriever_metrics: RetrieverMetrics
    generator_metrics: GeneratorMetrics
