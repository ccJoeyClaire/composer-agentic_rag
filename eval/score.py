"""RAGChecker scoring wrapper.

Encapsulates the external RAGChecker library so the rest of the eval
pipeline never imports ``ragchecker`` directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eval.run_config import EvalRunConfig, checker_output_path
from eval.types import CheckerInput, CheckerOutput, RunMetrics


@dataclass(frozen=True)
class RAGCheckerConfig:
    """Configuration for one RAGChecker evaluation run.

    Args:
        extractor_name:  Model name for claim extraction (RefChecker).
        checker_name:    Model name for entailment checking (RefChecker).
        batch_size_extractor: Batch size for the extractor pass.
        batch_size_checker:   Batch size for the checker pass.
        disable_joint_check:  When True, checks claims one-by-one (slower,
                                slightly more accurate).
    """

    extractor_name: str
    checker_name: str
    batch_size_extractor: int = 32
    batch_size_checker: int = 32
    disable_joint_check: bool = False


def load_checker_input(path: Path) -> CheckerInput:
    """Load a previously written RAGChecker input JSON file.

    Args:
        path: Path to a ``checking_inputs/*.json`` file.

    Returns:
        Parsed :class:`CheckerInput`.
    """
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_checker_output(path: Path) -> CheckerOutput:
    """Load a RAGChecker output JSON file.

    Args:
        path: Path to a ``checking_outputs/*.json`` file.

    Returns:
        Parsed :class:`CheckerOutput`.
    """
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_checker_output(output: CheckerOutput, path: Path) -> None:
    """Persist RAGChecker output to disk.

    Args:
        output: Parsed metrics dict.
        path:   Destination; parent dirs are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)


async def run_ragchecker(
    config: EvalRunConfig,
    checker_input: CheckerInput,
    checker_config: RAGCheckerConfig,
    *,
    output_path: Path | None = None,
) -> CheckerOutput:
    """Run RAGChecker on one assembled input and write the output.

    Args:
        config:          Eval run config (used for default output path).
        checker_input:   Assembled RAGChecker input.
        checker_config:  Model and batch settings.
        output_path:     Override destination; defaults to
                         :func:`~eval.run_config.checker_output_path`.

    Returns:
        Parsed :class:`CheckerOutput` metrics.

    Raises:
        ImportError: When ``ragchecker`` is not installed.
    """
    try:
        from ragchecker import RAGChecker, RAGResults
        from ragchecker.metrics import all_metrics
    except ImportError as exc:
        raise ImportError(
            "ragchecker is not installed. Run: pip install ragchecker"
        ) from exc

    rag_results = RAGResults.from_dict(checker_input)
    evaluator = RAGChecker(
        extractor_name=checker_config.extractor_name,
        checker_name=checker_config.checker_name,
        batch_size_extractor=checker_config.batch_size_extractor,
        batch_size_checker=checker_config.batch_size_checker,
        disable_joint_check=checker_config.disable_joint_check,
    )
    evaluator.evaluate(rag_results, all_metrics)

    output: CheckerOutput = {
        "overall_metrics": dict(rag_results.metrics.get("overall_metrics", {})),
        "retriever_metrics": dict(rag_results.metrics.get("retriever_metrics", {})),
        "generator_metrics": dict(rag_results.metrics.get("generator_metrics", {})),
    }

    resolved = output_path if output_path is not None else checker_output_path(config)
    write_checker_output(output, resolved)
    return output


def build_run_metrics(
    config: EvalRunConfig,
    checker_output: CheckerOutput,
    query_count: int,
) -> RunMetrics:
    """Annotate RAGChecker output with run identity for comparison tables.

    Args:
        config:          Eval run config.
        checker_output:  Parsed RAGChecker metrics.
        query_count:     Number of queries in this run.

    Returns:
        :class:`RunMetrics` ready for :mod:`eval.compare`.
    """
    return RunMetrics(
        run_id=config.run_id,
        doc_slug=config.doc_slug,
        profile_id=config.profile_id,
        arm=config.arm,
        query_count=query_count,
        overall=dict(checker_output.get("overall_metrics", {})),
        retriever=dict(checker_output.get("retriever_metrics", {})),
        generator=dict(checker_output.get("generator_metrics", {})),
    )
