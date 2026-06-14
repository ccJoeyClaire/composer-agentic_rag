"""Optional LLM-as-judge for agent answers.

BEIR datasets ship relevance labels (qrels) but no reference answer text, so
answer correctness cannot be scored by string match. This judge asks an LLM to
rate a candidate answer against a few gold passages on two axes:

* ``correct``  — does the answer actually address the query correctly?
* ``grounded`` — is the answer supported by the gold passages (not hallucinated)?

It is opt-in (``--judge``) because it costs an extra LLM call per query.
"""

from __future__ import annotations

import json
from typing import TypedDict

from llm.client import LLMClient

_MAX_PASSAGES = 3
_MAX_PASSAGE_CHARS = 1200

_JUDGE_PROMPT = """You are grading an assistant's answer to a question.

Question:
{query}

Assistant answer:
{answer}

Reference passages (known relevant evidence):
{passages}

Judge two things:
- "correct": true if the answer correctly and relevantly addresses the question.
- "grounded": true if the answer's claims are supported by the reference passages.

Return JSON only, no prose:
{{"correct": true|false, "grounded": true|false, "reason": "<one short sentence>"}}
"""


class JudgeVerdict(TypedDict):
    correct: bool
    grounded: bool
    reason: str


def _format_passages(gold_texts: list[str]) -> str:
    selected = [text[:_MAX_PASSAGE_CHARS] for text in gold_texts[:_MAX_PASSAGES]]
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(selected)) or "(none)"


async def judge_answer(
    llm: LLMClient,
    *,
    query: str,
    answer: str,
    gold_texts: list[str],
) -> JudgeVerdict:
    """Grade one answer; returns a conservative wrong/ungrounded verdict on error."""
    if not answer.strip():
        return JudgeVerdict(correct=False, grounded=False, reason="empty answer")

    prompt = _JUDGE_PROMPT.format(
        query=query,
        answer=answer,
        passages=_format_passages(gold_texts),
    )
    try:
        response = await llm.arequest_llm(
            [{"role": "user", "content": prompt}],
            json_output=True,
        )
        data = json.loads(response.content or "{}")
        return JudgeVerdict(
            correct=bool(data.get("correct", False)),
            grounded=bool(data.get("grounded", False)),
            reason=str(data.get("reason", "")),
        )
    except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as exc:
        return JudgeVerdict(correct=False, grounded=False, reason=f"judge error: {exc}")
