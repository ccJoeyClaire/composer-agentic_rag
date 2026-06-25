"""Shared types for the eval pipeline.

One canonical definition per concept.  Import from here; never duplicate.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# ---------------------------------------------------------------------------
# Arms (discriminated union of what generated the response)
# ---------------------------------------------------------------------------

EvalArm = Literal["direct_rag", "react", "crag", "self_rag", "crag_self_rag"]

DIRECT_RAG_ARM: EvalArm = "direct_rag"
AGENT_ARMS: tuple[EvalArm, ...] = ("react", "crag", "self_rag", "crag_self_rag")

# Profile IDs that participate in the matrix (§4.4 of blueprint).
EVAL_PROFILES: tuple[str, ...] = (
    "token",
    "baseline",
    "baseline_s2b",
    "baseline_predict_q",
    "baseline_hyde",
    "full",
)

# Question types admitted into RAGChecker evaluation (§4.6 of blueprint).
EVAL_QUESTION_TYPES = frozenset({"open_ended", "short_answer"})

# ---------------------------------------------------------------------------
# Gold (from gold_rubric.jsonl)
# ---------------------------------------------------------------------------


class GoldRecord(TypedDict):
    """One row from ``_eval_/datasets/Easy-Dataset/gold_rubric.jsonl``.

    Only the fields consumed by the eval pipeline are declared here.
    ``total=False`` sub-fields are intentionally omitted from *this* type;
    use ``RubricGold`` from ``_eval_.qa_eval.types`` for the full shape.
    """

    query_id: str
    question: str
    question_type: str  # one of EvalQuestionType literals
    answer: str         # gt_answer for RAGChecker (§4.1)
    source_doc_id: str
    source_path: str


# ---------------------------------------------------------------------------
# Infer results (one run of retrieve or agent)
# ---------------------------------------------------------------------------


class ContextChunk(TypedDict):
    """One retrieved chunk in RAGChecker's ``retrieved_context`` format."""

    doc_id: str   # chunk_id when available, else source filename
    text: str     # raw chunk content (no contextual header prefix)


class InferResult(TypedDict):
    """Output of one arm run for a single query."""

    query_id: str
    query: str
    response: str                      # generated answer
    retrieved_context: list[ContextChunk]
    arm: EvalArm
    profile_id: str


# ---------------------------------------------------------------------------
# RAGChecker wire format (input / output)
# ---------------------------------------------------------------------------


class CheckerInputRecord(TypedDict):
    """One record in the ``results`` array fed to RAGChecker."""

    query_id: str
    query: str
    gt_answer: str
    response: str
    retrieved_context: list[ContextChunk]


class CheckerInput(TypedDict):
    """Top-level RAGChecker input file."""

    results: list[CheckerInputRecord]


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


class CheckerOutput(TypedDict, total=False):
    """Parsed RAGChecker output for one run."""

    overall_metrics: OverallMetrics
    retriever_metrics: RetrieverMetrics
    generator_metrics: GeneratorMetrics


# ---------------------------------------------------------------------------
# Run-level results (metrics annotated with run identity)
# ---------------------------------------------------------------------------


class RunMetrics(TypedDict):
    """RAGChecker output merged with run identity for comparison tables."""

    run_id: str          # "{doc_slug}__{profile_id}__{arm}"
    doc_slug: str
    profile_id: str
    arm: EvalArm
    query_count: int
    overall: OverallMetrics
    retriever: RetrieverMetrics
    generator: GeneratorMetrics


# ---------------------------------------------------------------------------
# Delta comparison
# ---------------------------------------------------------------------------


class MetricDelta(TypedDict):
    """Absolute delta between two runs on a single metric."""

    metric: str        # e.g. "overall.f1"
    baseline_val: float
    candidate_val: float
    delta: float       # candidate - baseline
    improved: bool
