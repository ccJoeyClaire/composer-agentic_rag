"""
Predict questions per chunk at index time (#6 document augmentation).

Uses an LLM to generate questions each chunk can answer; stores them on
``metadata["predicted_questions"]`` and optionally appends them to ``embed_text``
so user queries align better with indexed vectors.

Run (from repo root):
  python -m rag.document_augmentation.predict_question
  python -m rag.document_augmentation.predict_question --live
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional

from llm.client import LLMClient

from ..base import BaseContextualEnricher, Chunk

DEFAULT_SYSTEM_PROMPT = """You generate questions that a text passage can answer.
Match the language of the passage. Return JSON only:
{"questions": ["question 1", "question 2"]}
Write concise, specific questions. Do not include answers."""

DEFAULT_USER_TEMPLATE = """Section: {heading_path}

Passage:
{content}

Generate {num_questions} questions this passage can answer."""

PREDICTED_QUESTIONS_KEY = "predicted_questions"
MAX_LLM_CONCURRENCY = 499


def _bounded_concurrency(max_concurrency: int) -> int:
    return max(1, min(max_concurrency, MAX_LLM_CONCURRENCY))


def parse_predicted_questions(raw: str) -> List[str]:
    """Parse LLM JSON output into a deduped question list."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        return []

    seen: set[str] = set()
    out: List[str] = []
    for item in questions:
        q = str(item).strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def build_questions_embed_suffix(questions: List[str]) -> str:
    """Format predicted questions for embedding augmentation."""
    if not questions:
        return ""
    lines = "\n".join(f"- {q}" for q in questions)
    return f"Possible questions:\n{lines}"


def append_questions_to_embed_text(embed_text: str, questions: List[str]) -> str:
    suffix = build_questions_embed_suffix(questions)
    if not suffix:
        return embed_text
    base = (embed_text or "").strip()
    if not base:
        return suffix
    return f"{base}\n\n{suffix}"


async def predict_questions_for_chunk(
    chunk: Chunk,
    llm: LLMClient,
    *,
    num_questions: int = 3,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
    temperature: float = 0.0,
) -> List[str]:
    content = (chunk.content or "").strip()
    if not content:
        return []

    meta = chunk.metadata or {}
    heading = meta.get("heading_path") or "—"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(
                heading_path=heading,
                content=content,
                num_questions=num_questions,
            ),
        },
    ]
    response = await llm.arequest_llm(
        messages,
        json_output=True,
        temperature=temperature,
    )
    return parse_predicted_questions(response.content or "")


async def augment_chunks_with_questions(
    chunks: List[Chunk],
    llm: LLMClient,
    *,
    num_questions: int = 3,
    append_to_embed_text: bool = True,
    max_concurrency: int = MAX_LLM_CONCURRENCY,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
    temperature: float = 0.0,
) -> List[Chunk]:
    if not chunks:
        return []

    limit = _bounded_concurrency(max_concurrency)
    sem = asyncio.Semaphore(limit)

    async def _predict_with_limit(chunk: Chunk) -> List[str]:
        async with sem:
            return await predict_questions_for_chunk(
                chunk,
                llm,
                num_questions=num_questions,
                system_prompt=system_prompt,
                user_template=user_template,
                temperature=temperature,
            )

    question_lists = await asyncio.gather(
        *(_predict_with_limit(chunk) for chunk in chunks)
    )

    return [
        _apply_predicted_questions(chunk, questions, append_to_embed_text)
        for chunk, questions in zip(chunks, question_lists)
    ]


def _apply_predicted_questions(
    chunk: Chunk,
    questions: List[str],
    append_to_embed_text: bool,
) -> Chunk:
    meta = dict(chunk.metadata or {})
    if questions:
        meta[PREDICTED_QUESTIONS_KEY] = questions
        if append_to_embed_text:
            base = meta.get("embed_text", chunk.content)
            meta["embed_text"] = append_questions_to_embed_text(base, questions)
    return Chunk(content=chunk.content, metadata=meta, score=chunk.score)


class PredictQuestionEnricher(BaseContextualEnricher):
    """Index-time enricher: LLM-generated preset questions per chunk."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        num_questions: int = 3,
        append_to_embed_text: bool = True,
        max_concurrency: int = MAX_LLM_CONCURRENCY,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_TEMPLATE,
        temperature: float = 0.0,
    ):
        self.llm = llm_client or LLMClient()
        self.num_questions = num_questions
        self.append_to_embed_text = append_to_embed_text
        self.max_concurrency = _bounded_concurrency(max_concurrency)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.temperature = temperature

    async def aenrich_for_index(
        self, chunks: List[Chunk], *, source: str = ""
    ) -> List[Chunk]:
        if source:
            chunks = [
                Chunk(
                    content=chunk.content,
                    metadata={**dict(chunk.metadata or {}), "source": source},
                    score=chunk.score,
                )
                for chunk in chunks
            ]
        return await augment_chunks_with_questions(
            chunks,
            self.llm,
            num_questions=self.num_questions,
            append_to_embed_text=self.append_to_embed_text,
            max_concurrency=self.max_concurrency,
            system_prompt=self.system_prompt,
            user_template=self.user_template,
            temperature=self.temperature,
        )

    async def aenrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        return list(chunks)


async def _demo_main() -> None:
    """Offline parse demo or live LLM enrichment."""
    import argparse
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Predict-question enrichment demo.")
    parser.add_argument("--live", action="store_true", help="Call LLM for one chunk")
    args = parser.parse_args()

    fixture = '{"questions": ["What is RAG?", "How does retrieval help LLMs?"]}'
    parsed = parse_predicted_questions(fixture)
    print("=== parse_predicted_questions (offline) ===")
    print(parsed)

    if not args.live:
        return

    if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("Missing LLM_API_KEY for --live.", file=sys.stderr)
        sys.exit(1)

    chunk = Chunk(
        content="RAG retrieves documents before the LLM answers.",
        metadata={"heading_path": "RAG > Overview"},
    )
    enricher = PredictQuestionEnricher(num_questions=2)
    enriched = await enricher.aenrich_for_index([chunk], source="demo.md")
    meta = enriched[0].metadata or {}
    print("\n=== PredictQuestionEnricher.aenrich_for_index ===")
    print(f"  questions={meta.get(PREDICTED_QUESTIONS_KEY)}")
    print(f"  embed_text preview={str(meta.get('embed_text', ''))[:200]}...")


def _main() -> None:
    import asyncio

    asyncio.run(_demo_main())


if __name__ == "__main__":
    _main()
