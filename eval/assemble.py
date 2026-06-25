"""Assemble InferResult batch + gold → RAGChecker input JSON.

This is the bridge between the inference pipeline and RAGChecker.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.run_config import EvalRunConfig, checker_input_path
from eval.types import CheckerInput, CheckerInputRecord, GoldRecord, InferResult


def build_checker_record(
    gold: GoldRecord,
    infer: InferResult,
) -> CheckerInputRecord:
    """Merge one gold record with its infer result into a RAGChecker record.

    Args:
        gold:  Gold record supplying query_id, question, and gt answer.
        infer: Infer result supplying response and retrieved context.

    Returns:
        One :class:`CheckerInputRecord` ready for RAGChecker.
    """
    return CheckerInputRecord(
        query_id=gold["query_id"],
        query=gold["question"],
        gt_answer=gold["answer"],
        response=infer["response"],
        retrieved_context=infer["retrieved_context"],
    )


def build_checker_input(
    gold_records: list[GoldRecord],
    infer_results: list[InferResult],
) -> CheckerInput:
    """Merge a full gold set with a full infer batch.

    Records are matched by ``query_id``.  Missing infer results raise
    :class:`KeyError` so silent gaps are caught before scoring.

    Args:
        gold_records:  Filtered gold (same order as the infer batch).
        infer_results: One :class:`InferResult` per gold record.

    Returns:
        Top-level :class:`CheckerInput` for RAGChecker.
    """
    infer_by_id: dict[str, InferResult] = {r["query_id"]: r for r in infer_results}
    records: list[CheckerInputRecord] = []
    for gold in gold_records:
        infer = infer_by_id[gold["query_id"]]
        records.append(build_checker_record(gold, infer))
    return CheckerInput(results=records)


def write_checker_input(
    checker_input: CheckerInput,
    path: Path,
) -> None:
    """Write a :class:`CheckerInput` to disk as pretty JSON.

    Args:
        checker_input: Assembled input for one run.
        path:            Destination file path; parent dirs are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(checker_input, fh, ensure_ascii=False, indent=2)


def assemble_and_write(
    config: EvalRunConfig,
    gold_records: list[GoldRecord],
    infer_results: list[InferResult],
    *,
    output_path: Path | None = None,
) -> Path:
    """Build checker input and write to the canonical run path.

    Args:
        config:        Run config (used for default output path).
        gold_records:  Filtered gold records.
        infer_results: Infer batch for this run.
        output_path:   Override destination; defaults to
                       :func:`~eval.run_config.checker_input_path`.

    Returns:
        Path where the file was written.
    """
    resolved = output_path if output_path is not None else checker_input_path(config)
    checker_input = build_checker_input(gold_records, infer_results)
    write_checker_input(checker_input, resolved)
    return resolved
