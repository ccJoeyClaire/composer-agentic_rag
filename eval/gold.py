"""Load and filter gold records from ``gold_rubric.jsonl``.

Ownership: the canonical gold lives in
``_eval_/datasets/Easy-Dataset/gold_rubric.jsonl``.
This module reads it read-only; never writes back.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.types import EVAL_QUESTION_TYPES, GoldRecord

# Default path relative to repo root.
_DEFAULT_GOLD_PATH = Path(__file__).resolve().parents[1] / (
    "_eval_/datasets/Easy-Dataset/gold_rubric.jsonl"
)


def load_gold_records(path: Path | None = None) -> list[GoldRecord]:
    """Load every row from ``gold_rubric.jsonl`` as a :class:`GoldRecord`.

    Args:
        path: Explicit JSONL path.  ``None`` uses the default Easy-Dataset gold.

    Returns:
        All rows in file order; no filtering applied.
    """
    resolved = path if path is not None else _DEFAULT_GOLD_PATH
    records: list[GoldRecord] = []
    with resolved.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                GoldRecord(
                    query_id=str(raw["query_id"]),
                    question=str(raw["question"]),
                    question_type=str(raw.get("question_type", "")),
                    answer=str(raw["answer"]),
                    source_doc_id=str(raw.get("source_doc_id", "")),
                    source_path=str(raw.get("source_path", "")),
                )
            )
    return records


def filter_for_eval(records: list[GoldRecord]) -> list[GoldRecord]:
    """Keep only question types admitted by the eval blueprint (§4.6).

    Admitted: ``open_ended``, ``short_answer``.
    Excluded: ``multiple_choice``, ``single_choice``, ``true_false``.

    Args:
        records: Full gold loaded by :func:`load_gold_records`.

    Returns:
        Filtered subset in original order.
    """
    return [r for r in records if r["question_type"] in EVAL_QUESTION_TYPES]


def load_eval_gold(path: Path | None = None) -> list[GoldRecord]:
    """Convenience: load + filter in one call.

    Args:
        path: Forwarded to :func:`load_gold_records`.

    Returns:
        Filtered gold records ready for the eval pipeline.
    """
    return filter_for_eval(load_gold_records(path))
