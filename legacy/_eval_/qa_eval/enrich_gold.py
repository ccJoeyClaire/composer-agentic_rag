"""Enrich Easy Dataset QA export with checklist rubrics (scheme B, step 1).

Reads draft ``question`` + ``correctAnswer`` pairs, uses an LLM to extract
``correct_incidents`` and ``complete_key_points`` grounded in the source document.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from _eval_.paths import REPO_ROOT
from _eval_.qa_eval.easy_export import (
    EasyExportBundle,
    assign_query_ids,
    load_easy_export,
    normalize_question_type,
    slugify_doc_id,
    write_gold_rubric,
)
from _eval_.qa_eval.types import EasyExportRow, QuestionType, RubricGold
from llm.client import LLMClient

_DEFAULT_EXPORT = (
    REPO_ROOT
    / "_eval_"
    / "datasets"
    / "Easy-Dataset"
    / "eval-datasets-1781929888288.jsonl"
)
_DEFAULT_SOURCE = (
    REPO_ROOT
    / "_eval_"
    / "datasets"
    / "Easy-Dataset"
    / "file"
    / "工程技术：在智能体优先的世界中利用 Codex.md"
)
_DEFAULT_OUT = (
    REPO_ROOT / "_eval_" / "datasets" / "Easy-Dataset" / "gold_rubric.jsonl"
)

_MAX_SOURCE_CHARS = 48_000
_DEFAULT_CONCURRENCY = 5

_ENRICH_SYSTEM = """You extract evaluation rubrics from QA pairs for RAG benchmarking.
Output JSON only. Each rubric item must be atomic, objectively checkable, and grounded in the source document."""

_ENRICH_USER = """Source document:
------ START ------
{source}
------ END ------

Question type: {question_type}
Question:
{question}

Draft reference answer (from dataset author):
{draft_answer}

Extract two checklists for scoring candidate answers with temperature=0 binary judges:

1. correct_incidents — core factual claims. If ANY is missing, the answer is materially wrong.
   - 3 to 8 items for open_ended; 1 to 3 for short_answer / choice / true_false
   - Each item one verifiable fact or conclusion, no compound "and/or" bundles

2. complete_key_points — additional points required for a fully complete answer.
   - 3 to 8 items for open_ended; 0 to 2 for short_answer / choice / true_false
   - Must not duplicate incidents verbatim; focus on coverage, nuance, structure, trade-offs

Also return a short ``answer`` field: 2-4 sentence canonical reference (optional compression of draft).

Return JSON:
{{
  "answer": "...",
  "correct_incidents": ["...", "..."],
  "complete_key_points": ["...", "..."]
}}
"""


def _filter_rows(
    rows: list[tuple[str, EasyExportRow]],
    *,
    question_types: set[QuestionType] | None,
) -> list[tuple[str, EasyExportRow]]:
    if not question_types:
        return rows
    return [
        (query_id, row)
        for query_id, row in rows
        if normalize_question_type(row.get("questionType", "")) in question_types
    ]


def _truncate_source(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n... [truncated, {len(text)} chars total]"


def _parse_enrichment_payload(raw: str) -> dict[str, object]:
    data = json.loads(raw or "{}")
    incidents = data.get("correct_incidents") or []
    key_points = data.get("complete_key_points") or []
    if not isinstance(incidents, list) or not isinstance(key_points, list):
        raise ValueError("rubric lists must be arrays")
    return {
        "answer": str(data.get("answer", "")),
        "correct_incidents": [str(item).strip() for item in incidents if str(item).strip()],
        "complete_key_points": [str(item).strip() for item in key_points if str(item).strip()],
    }


async def enrich_one(
    llm: LLMClient,
    *,
    query_id: str,
    row: EasyExportRow,
    source_text: str,
    doc_id: str,
    source_path: Path,
    temperature: float,
) -> RubricGold:
    """Build one :class:`RubricGold` via LLM rubric extraction."""
    question_type = normalize_question_type(row.get("questionType", ""))
    prompt = _ENRICH_USER.format(
        source=_truncate_source(source_text, _MAX_SOURCE_CHARS),
        question_type=question_type,
        question=row.get("question", ""),
        draft_answer=row.get("correctAnswer", ""),
    )
    response = await llm.arequest_llm(
        [
            {"role": "system", "content": _ENRICH_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        json_output=True,
        temperature=temperature,
    )
    parsed = _parse_enrichment_payload(response.content or "{}")
    incidents = parsed["correct_incidents"]
    key_points = parsed["complete_key_points"]
    if not incidents:
        raise ValueError(f"{query_id}: empty correct_incidents")

    return RubricGold(
        query_id=query_id,
        question=row.get("question", ""),
        question_type=question_type,
        draft_answer=row.get("correctAnswer", ""),
        answer=str(parsed["answer"] or row.get("correctAnswer", "")),
        correct_incidents=incidents,
        complete_key_points=key_points,
        source_doc_id=doc_id,
        source_path=str(source_path),
    )


async def enrich_bundle(
    bundle: EasyExportBundle,
    *,
    llm: LLMClient,
    concurrency: int,
    temperature: float,
    question_types: set[QuestionType] | None = None,
) -> list[RubricGold]:
    """Enrich all rows in ``bundle`` with bounded concurrency."""
    source_text = bundle.source_path.read_text(encoding="utf-8")
    rows = _filter_rows(bundle.rows, question_types=question_types)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[RubricGold] = []

    async def _one(query_id: str, row: EasyExportRow) -> RubricGold:
        async with semaphore:
            return await enrich_one(
                llm,
                query_id=query_id,
                row=row,
                source_text=source_text,
                doc_id=bundle.doc_id,
                source_path=bundle.source_path,
                temperature=temperature,
            )

    tasks = [_one(qid, row) for qid, row in rows]
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    results.sort(key=lambda item: item["query_id"])
    return results


def build_bundle(
    *,
    export_path: Path,
    source_path: Path,
    doc_id: str | None = None,
) -> EasyExportBundle:
    resolved_doc_id = doc_id or slugify_doc_id(source_path.stem)
    rows = load_easy_export(export_path)
    keyed = assign_query_ids(rows, doc_id=resolved_doc_id)
    return EasyExportBundle(doc_id=resolved_doc_id, source_path=source_path, rows=keyed)


async def run_enrich(
    *,
    export_path: Path,
    source_path: Path,
    out_path: Path,
    doc_id: str | None,
    question_types: set[QuestionType] | None,
    concurrency: int,
    temperature: float,
) -> list[RubricGold]:
    bundle = build_bundle(export_path=export_path, source_path=source_path, doc_id=doc_id)
    llm = LLMClient()
    records = await enrich_bundle(
        bundle,
        llm=llm,
        concurrency=concurrency,
        temperature=temperature,
        question_types=question_types,
    )
    write_gold_rubric(out_path, records)
    return records


def _parse_question_types(raw: list[str] | None) -> set[QuestionType] | None:
    if not raw:
        return None
    parsed = {normalize_question_type(item) for item in raw}
    return {item for item in parsed if item != "unknown"}


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Enrich Easy Dataset export with rubric checklists (scheme B step 1)."
    )
    parser.add_argument("--export", type=Path, default=_DEFAULT_EXPORT)
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--doc-id", default=None, help="Corpus doc id slug (default: from source filename)")
    parser.add_argument(
        "--question-type",
        action="append",
        dest="question_types",
        help="Repeatable filter, e.g. open_ended. Default: all rows.",
    )
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("EVAL_LLM_CONCURRENCY", _DEFAULT_CONCURRENCY)))
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    records = asyncio.run(
        run_enrich(
            export_path=args.export,
            source_path=args.source,
            out_path=args.out,
            doc_id=args.doc_id,
            question_types=_parse_question_types(args.question_types),
            concurrency=args.concurrency,
            temperature=args.temperature,
        )
    )
    print(f"wrote {len(records)} rubric records -> {args.out}")


if __name__ == "__main__":
    main()
