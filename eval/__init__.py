"""Eval pipeline: Easy Dataset gold → RAGChecker metrics.

See ``eval/eval_blueprint.md`` for the full design document.

Quick start::

    python -m eval.run index --doc-slug codex
    python -m eval.run infer  --doc-slug codex --profile baseline --arm direct_rag
    python -m eval.run score   --doc-slug codex --profile baseline --arm direct_rag \\
        --extractor <TBD> --checker <TBD>
"""

from eval.assemble import assemble_and_write, build_checker_input, build_checker_record
from eval.compare import compute_deltas, load_all_run_metrics, load_run_metrics
from eval.gold import filter_for_eval, load_eval_gold, load_gold_records
from eval.infer import AgentInferArm, BaseInferArm, DirectRagArm
from eval.index import IndexResult, index_profiles
from eval.pipeline import (
    run_assemble_stage,
    run_full_matrix,
    run_index_stage,
    run_infer_stage,
    run_score_stage,
)
from eval.run_config import (
    EvalRunConfig,
    build_run_matrix,
    checker_input_path,
    checker_output_path,
    index_configs_for_matrix,
    infer_output_path,
)
from eval.score import RAGCheckerConfig, build_run_metrics, run_ragchecker
from eval.types import (
    AGENT_ARMS,
    DIRECT_RAG_ARM,
    EVAL_PROFILES,
    EVAL_QUESTION_TYPES,
    CheckerInput,
    CheckerInputRecord,
    CheckerOutput,
    ContextChunk,
    EvalArm,
    GeneratorMetrics,
    GoldRecord,
    InferResult,
    MetricDelta,
    OverallMetrics,
    RetrieverMetrics,
    RunMetrics,
)

__all__ = [
    # Constants
    "AGENT_ARMS",
    "DIRECT_RAG_ARM",
    "EVAL_PROFILES",
    "EVAL_QUESTION_TYPES",
    # Types
    "CheckerInput",
    "CheckerInputRecord",
    "CheckerOutput",
    "ContextChunk",
    "EvalArm",
    "GeneratorMetrics",
    "GoldRecord",
    "InferResult",
    "MetricDelta",
    "OverallMetrics",
    "RetrieverMetrics",
    "RunMetrics",
    # Config
    "EvalRunConfig",
    "RAGCheckerConfig",
    "build_run_matrix",
    "index_configs_for_matrix",
    "checker_input_path",
    "checker_output_path",
    "infer_output_path",
    # Gold
    "load_eval_gold",
    "load_gold_records",
    "filter_for_eval",
    # Index
    "IndexResult",
    "index_profiles",
    # Infer
    "BaseInferArm",
    "DirectRagArm",
    "AgentInferArm",
    # Assemble
    "build_checker_record",
    "build_checker_input",
    "assemble_and_write",
    # Score
    "run_ragchecker",
    "build_run_metrics",
    # Compare
    "compute_deltas",
    "load_run_metrics",
    "load_all_run_metrics",
    # Pipeline
    "run_index_stage",
    "run_infer_stage",
    "run_assemble_stage",
    "run_score_stage",
    "run_full_matrix",
]
