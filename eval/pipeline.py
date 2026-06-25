"""Batch orchestration: index → infer → assemble → score.

Each stage function accepts an explicit matrix and gold so individual
stages can also be run standalone via :mod:`eval.run`.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.assemble import assemble_and_write, build_checker_input
from eval.gold import load_eval_gold
from eval.infer.agent import AgentInferArm
from eval.infer.base import BaseInferArm
from eval.infer.direct_rag import DirectRagArm
from eval.index import IndexResult, index_profiles
from eval.run_config import EvalRunConfig, build_run_matrix, infer_output_path
from eval.score import RAGCheckerConfig, build_run_metrics, run_ragchecker
from eval.types import GoldRecord, InferResult


def _make_arm(config: EvalRunConfig) -> BaseInferArm:
    """Instantiate the correct infer arm for a run config."""
    if config.is_direct_rag:
        return DirectRagArm(config)
    return AgentInferArm(config)


def _write_infer_results(results: list[InferResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)


def _load_infer_results(path: Path) -> list[InferResult]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


async def run_index_stage(
    matrix: list[EvalRunConfig],
    source_path: Path,
) -> list[IndexResult]:
    """Stage 1: index the source document for all unique profiles.

    Args:
        matrix:      Full run matrix.
        source_path: Path to the source document.

    Returns:
        One :class:`IndexResult` per unique profile.
    """
    return await index_profiles(matrix, source_path)


async def run_infer_stage(
    config: EvalRunConfig,
    gold_records: list[GoldRecord],
) -> list[InferResult]:
    """Stage 2: run one arm for all gold queries and persist raw results.

    Args:
        config:       Run config identifying profile and arm.
        gold_records: Filtered gold records to evaluate.

    Returns:
        Infer results written to ``eval/runs/{arm}/{run_id}.json``.
    """
    arm = _make_arm(config)
    try:
        query_pairs = [(g["query_id"], g["question"]) for g in gold_records]
        results = await arm.arun_batch(query_pairs)
    finally:
        await arm.aclose()

    out_path = infer_output_path(config)
    _write_infer_results(results, out_path)
    return results


async def run_assemble_stage(
    config: EvalRunConfig,
    gold_records: list[GoldRecord],
    infer_results: list[InferResult] | None = None,
) -> Path:
    """Stage 3: merge infer results with gold → RAGChecker input JSON.

    Args:
        config:        Run config.
        gold_records:  Filtered gold records.
        infer_results: Pre-loaded results; loaded from disk when ``None``.

    Returns:
        Path to the written checker input file.
    """
    if infer_results is None:
        infer_results = _load_infer_results(infer_output_path(config))
    return assemble_and_write(config, gold_records, infer_results)


async def run_score_stage(
    config: EvalRunConfig,
    checker_config: RAGCheckerConfig,
    gold_records: list[GoldRecord],
) -> Path:
    """Stage 4: run RAGChecker on the assembled input.

    Args:
        config:          Run config.
        checker_config:  RAGChecker model settings.
        gold_records:    Used only for query_count annotation.

    Returns:
        Path to the written checker output file.
    """
    from eval.run_config import checker_output_path

    infer_results = _load_infer_results(infer_output_path(config))
    checker_input = build_checker_input(gold_records, infer_results)
    output = await run_ragchecker(config, checker_input, checker_config)
    build_run_metrics(config, output, query_count=len(gold_records))
    return checker_output_path(config)


async def run_full_matrix(
    doc_slug: str,
    source_path: Path,
    checker_config: RAGCheckerConfig,
    *,
    gold_path: Path | None = None,
    profiles: tuple[str, ...] | None = None,
) -> list[EvalRunConfig]:
    """Run the complete eval pipeline for one document.

    Executes index → infer → assemble → score for every run in the matrix.

    Args:
        doc_slug:        Short document identifier.
        source_path:     Path to the source document.
        checker_config:  RAGChecker model settings.
        gold_path:       Override gold JSONL path.
        profiles:        Override profile list; defaults to blueprint set.

    Returns:
        The run matrix that was executed.
    """
    from eval.types import EVAL_PROFILES

    matrix = build_run_matrix(
        doc_slug,
        profiles=profiles if profiles is not None else EVAL_PROFILES,
    )
    gold_records = load_eval_gold(gold_path)

    await run_index_stage(matrix, source_path)

    for cfg in matrix:
        await run_infer_stage(cfg, gold_records)
        await run_assemble_stage(cfg, gold_records)
        await run_score_stage(cfg, checker_config, gold_records)

    return matrix
