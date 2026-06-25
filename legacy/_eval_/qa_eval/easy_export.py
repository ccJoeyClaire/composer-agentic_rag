"""Load Easy Dataset eval export JSONL."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from _eval_.qa_eval.types import EasyExportRow, QuestionType, RubricGold

_DEFAULT_DOC_ID = "codex-agentic"


def slugify_doc_id(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower())
    return re.sub(r"-+", "-", slug).strip("-") or _DEFAULT_DOC_ID


def normalize_question_type(raw: str) -> QuestionType:
    value = (raw or "").strip().lower()
    if value in {
        "open_ended",
        "short_answer",
        "multiple_choice",
        "single_choice",
        "true_false",
    }:
        return value  # type: ignore[return-value]
    return "unknown"


def load_easy_export(path: Path) -> list[EasyExportRow]:
    """Load Easy Dataset eval export as typed rows."""
    rows: list[EasyExportRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                EasyExportRow(
                    questionType=str(raw.get("questionType", "")),
                    question=str(raw.get("question", "")),
                    options=str(raw.get("options", "")),
                    correctAnswer=str(raw.get("correctAnswer", "")),
                    tags=str(raw.get("tags", "")),
                )
            )
    return rows


def assign_query_ids(
    rows: list[EasyExportRow],
    *,
    doc_id: str,
) -> list[tuple[str, EasyExportRow]]:
    """Attach stable ``query_id`` values in export order."""
    return [(f"{doc_id}::q{index:04d}", row) for index, row in enumerate(rows)]


def write_gold_rubric(path: Path, records: list[RubricGold]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_gold_rubric(path: Path) -> list[RubricGold]:
    records: list[RubricGold] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class EasyExportBundle:
    """Export rows plus corpus metadata for one source document."""

    doc_id: str
    source_path: Path
    rows: list[tuple[str, EasyExportRow]]
