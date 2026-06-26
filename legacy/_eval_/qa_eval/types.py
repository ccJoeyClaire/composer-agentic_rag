"""Types for QA rubric enrichment and scoring."""

from __future__ import annotations

from typing import Literal, TypedDict

QuestionType = Literal[
    "open_ended",
    "short_answer",
    "multiple_choice",
    "single_choice",
    "true_false",
    "unknown",
]

EasyQuestionType = QuestionType


class EasyExportRow(TypedDict, total=False):
    """One row from Easy Dataset eval export JSONL."""

    questionType: str
    question: str
    options: str
    correctAnswer: str
    tags: str


class RubricGold(TypedDict, total=False):
    """Enriched gold record with checklist rubric for fair scoring."""

    query_id: str
    question: str
    question_type: QuestionType
    draft_answer: str
    answer: str
    correct_incidents: list[str]
    complete_key_points: list[str]
    source_doc_id: str
    source_path: str


class RubricItemScore(TypedDict):
    item: str
    score: int


class RubricScoreResult(TypedDict):
    query_id: str
    profile: str
    candidate_answer: str
    incident_scores: list[RubricItemScore]
    key_point_scores: list[RubricItemScore]
    correct_rate: float
    complete_rate: float
    reason: str
