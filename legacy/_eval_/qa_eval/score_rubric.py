"""Score candidate answers against rubric checklists (scheme B, step 2).

Uses temperature=0 and one LLM call per rubric item, returning 0 or 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from _eval_.paths import REPO_ROOT
from _eval_.qa_eval.easy_export import load_gold_rubric
from _eval_.qa_eval.types import RubricGold, RubricItemScore, RubricScoreResult
from llm.client import LLMClient

_DEFAULT_GOLD = REPO_ROOT / "_eval_" / "datasets" / "Easy-Dataset" / "gold_rubric.jsonl"
_DEFAULT_CONCURRENCY = 8

_ITEM_JUDGE_SYSTEM = """You are a strict binary grader for RAG answer evaluation.
Return JSON only: {"score": 0} or {"score": 1}. No explanation."""

_ITEM_JUDGE_USER = """Question:
{question}

Rubric item (the candidate must express or clearly imply this):
{item}

Candidate answer:
{candidate}

Does the candidate answer satisfy this rubric item?
- score 1: clearly present or equivalent
- score 0: missing, wrong, or too vague

Return JSON: {{"score": 0}} or {{"score": 1}}
"""


@dataclass(frozen=True)
class CandidateAnswer:
    query_id: str
    profile: str
    answer: str


def load_candidates(path: Path) -> list[CandidateAnswer]:
    """Load candidate answers JSONL: ``{query_id, profile, answer}``."""
    rows: list[CandidateAnswer] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                CandidateAnswer(
                    query_id=str(raw["query_id"]),
                    profile=str(raw.get("profile", "unknown")),
                    answer=str(raw.get("answer", "")),
                )
            )
    return rows


def _rate(items: list[RubricItemScore]) -> float:
    if not items:
        return 1.0
    return sum(item["score"] for item in items) / len(items)


async def score_one_item(
    llm: LLMClient,
    *,
    question: str,
    item: str,
    candidate: str,
    temperature: float,
) -> int:
    """Return 0 or 1 for one rubric item."""
    if not candidate.strip():
        return 0

    response = await llm.arequest_llm(
        [
            {"role": "system", "content": _ITEM_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _ITEM_JUDGE_USER.format(
                    question=question,
                    item=item,
                    candidate=candidate,
                ),
            },
        ],
        json_output=True,
        temperature=temperature,
    )
    data = json.loads(response.content or "{}")
    return 1 if int(data.get("score", 0)) == 1 else 0


async def score_candidate(
    llm: LLMClient,
    *,
    gold: RubricGold,
    candidate: CandidateAnswer,
    temperature: float,
) -> RubricScoreResult:
    """Score one candidate against all incidents and key points."""
    question = gold.get("question", "")
    incidents = gold.get("correct_incidents") or []
    key_points = gold.get("complete_key_points") or []

    incident_scores: list[RubricItemScore] = []
    for item in incidents:
        score = await score_one_item(
            llm,
            question=question,
            item=item,
            candidate=candidate.answer,
            temperature=temperature,
        )
        incident_scores.append(RubricItemScore(item=item, score=score))

    key_point_scores: list[RubricItemScore] = []
    for item in key_points:
        score = await score_one_item(
            llm,
            question=question,
            item=item,
            candidate=candidate.answer,
            temperature=temperature,
        )
        key_point_scores.append(RubricItemScore(item=item, score=score))

    correct_rate = _rate(incident_scores)
    complete_rate = _rate(key_point_scores)
    return RubricScoreResult(
        query_id=candidate.query_id,
        profile=candidate.profile,
        candidate_answer=candidate.answer,
        incident_scores=incident_scores,
        key_point_scores=key_point_scores,
        correct_rate=correct_rate,
        complete_rate=complete_rate,
        reason="",
    )


async def score_all(
    gold_records: list[RubricGold],
    candidates: list[CandidateAnswer],
    *,
    concurrency: int,
    temperature: float,
) -> list[RubricScoreResult]:
    gold_by_id = {record["query_id"]: record for record in gold_records}
    llm = LLMClient()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[RubricScoreResult] = []

    async def _one(candidate: CandidateAnswer) -> RubricScoreResult:
        gold = gold_by_id.get(candidate.query_id)
        if gold is None:
            return RubricScoreResult(
                query_id=candidate.query_id,
                profile=candidate.profile,
                candidate_answer=candidate.answer,
                incident_scores=[],
                key_point_scores=[],
                correct_rate=0.0,
                complete_rate=0.0,
                reason=f"unknown query_id: {candidate.query_id}",
            )
        async with semaphore:
            return await score_candidate(
                llm,
                gold=gold,
                candidate=candidate,
                temperature=temperature,
            )

    tasks = [_one(candidate) for candidate in candidates]
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    results.sort(key=lambda row: (row["profile"], row["query_id"]))
    return results


def summarize_by_profile(results: list[RubricScoreResult]) -> dict[str, dict[str, float]]:
    """Aggregate mean correct/complete rates per profile."""
    buckets: dict[str, list[RubricScoreResult]] = {}
    for row in results:
        buckets.setdefault(row["profile"], []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for profile, rows in sorted(buckets.items()):
        n = len(rows)
        summary[profile] = {
            "num_scored": float(n),
            "correct_rate": sum(r["correct_rate"] for r in rows) / n if n else 0.0,
            "complete_rate": sum(r["complete_rate"] for r in rows) / n if n else 0.0,
        }
    return summary


def write_score_results(path: Path, results: list[RubricScoreResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_score_results(path: Path) -> list[RubricScoreResult]:
    """Load rubric score rows written by :func:`write_score_results`."""
    rows: list[RubricScoreResult] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Score candidate answers with T=0 per-rubric-item binary judges."
    )
    parser.add_argument("--gold", type=Path, default=_DEFAULT_GOLD)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    gold_records = load_gold_rubric(args.gold)
    candidates = load_candidates(args.candidates)
    results = asyncio.run(
        score_all(
            gold_records,
            candidates,
            concurrency=args.concurrency,
            temperature=args.temperature,
        )
    )

    out_path = args.out or args.candidates.with_name("rubric_scores.jsonl")
    write_score_results(out_path, results)
    summary = summarize_by_profile(results)

    print(f"scored {len(results)} candidates -> {out_path}\n")
    print("| profile | n | correct_rate | complete_rate |")
    print("| --- | ---: | ---: | ---: |")
    for profile, metrics in summary.items():
        print(
            f"| {profile} | {int(metrics['num_scored'])} | "
            f"{metrics['correct_rate']:.3f} | {metrics['complete_rate']:.3f} |"
        )


if __name__ == "__main__":
    main()
