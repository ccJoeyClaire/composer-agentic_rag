
from __future__ import annotations

from typing import Literal, TypedDict


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class GoldQAPair(TypedDict):
    """Minimal gold QA used by infer / RAGChecker assembly."""

    query_id: str
    gold_question: str  # retrieval-ready question
    gt_answer: str      # concise, high-information reference answer


class AgentRunnerConfig(TypedDict, total=False):
    """Agent-side settings for one eval runner. Fields TBD."""

    pass


class RagRunnerConfig(TypedDict, total=False):
    """RAG index / retrieval settings for one eval runner. Fields TBD."""

    pass


class PipelineRunner(TypedDict):
    """One infer arm configuration (maps to a ``run_id`` in the eval matrix)."""

    runner_id: str
    agent_config: AgentRunnerConfig
    rag_config: RagRunnerConfig


class RetrievedChunk(TypedDict):
    """One chunk in RAGChecker's ``retrieved_context`` wire format."""

    doc_id: str  # chunk_id when available, else source filename
    text: str    # raw chunk content (no contextual header prefix)


class InferenceRecord(TypedDict):
    """Per-query infer output only.

    Runner identity (``runner_id``, profile, arm, configs) lives in
    :class:`PipelineRunner` and is joined at assembly time — not duplicated here.
    Infer JSON files are already scoped to one run (path encodes ``run_id``).
    """

    query_id: str
    generator_response: str
    retrieved_context: list[RetrievedChunk]


class RagCheckerSample(TypedDict):
    """One ``results[]`` row: gold + one runner's inference for a single query.

    Serialized to RAGChecker as ``query`` / ``response`` / ``gt_answer``;
    internal names stay domain-specific (see :mod:`eval.assemble`).
    """

    runner_id: str
    query_id: str
    gold_question: str
    gt_answer: str
    generator_response: str
    retrieved_context: list[RetrievedChunk]


class RagCheckerInput(TypedDict):
    """Top-level RAGChecker input file."""

    results: list[RagCheckerSample]


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


class RagCheckerRunMetrics(TypedDict, total=False):
    """Aggregate metrics for one ``evaluate()`` call over a full ``results`` batch.

    RAGChecker averages across all samples; this is **not** a per-query list.
    """

    overall_metrics: OverallMetrics
    retriever_metrics: RetrieverMetrics
    generator_metrics: GeneratorMetrics