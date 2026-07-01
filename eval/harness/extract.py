"""Map infer artifacts + gold into ``CheckerSample`` rows."""

from __future__ import annotations

import re

from agent.core.metadata.base import RETRIEVED_CONTEXT_KEY
from eval.base import (
    AgentInferArtifact,
    CheckerContextChunk,
    CheckerSample,
    GoldSample,
    RagInferArtifact,
)

_DOC_ID_SEP = "|"
_THINK_ACTION_FENCE = re.compile(
    r"```\s*\nThink:.*?```\s*\n?",
    re.DOTALL | re.IGNORECASE,
)


def checker_doc_id(*, source: str, heading_path: str) -> str:
    """Stable chunk id for RAGChecker: ``source|heading_path``."""
    return f"{source}{_DOC_ID_SEP}{heading_path}"


def strip_think_action_fences(text: str) -> str:
    """Drop the leading fenced Think/Action block from agent final answers."""
    return _THINK_ACTION_FENCE.sub("", text, count=1).strip()


def agent_artifact_to_checker_sample(
    artifact: AgentInferArtifact,
    gold: GoldSample,
) -> CheckerSample:
    """Build one checker row from an agent infer dump + gold."""
    run = artifact["run"]
    metadata = run.get("metadata") or {}
    retrieved_context: list[CheckerContextChunk] = []
    for item in metadata.get(RETRIEVED_CONTEXT_KEY) or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "")
        heading_path = str(item.get("heading_path") or "")
        content = str(item.get("content") or "")
        if content.startswith("Document:"):
            _header, _, body = content.partition("\n\n")
            text = body if body else content
        else:
            text = content
        retrieved_context.append(
            CheckerContextChunk(
                doc_id=checker_doc_id(source=source, heading_path=heading_path),
                text=text,
            )
        )

    final = run.get("final_message") or {}
    generator_response = strip_think_action_fences(str(final.get("content") or ""))

    return CheckerSample(
        query_id=artifact["query_id"],
        gold_question=gold["user_question"],
        gt_answer=gold["gt_answer"],
        generator_response=generator_response,
        retrieved_context=retrieved_context,
    )


def rag_artifact_to_checker_sample(
    artifact: RagInferArtifact,
    gold: GoldSample,
) -> CheckerSample:
    """Build one checker row from a RAG infer dump + gold."""
    stages = artifact["trace"].get("stages") or {}
    retrieved_context: list[CheckerContextChunk] = []
    for item in stages.get("final") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "")
        heading_path = str(item.get("heading_path") or "")
        content = str(item.get("content") or "")
        header = item.get("contextual_header")
        if isinstance(header, str) and content.startswith(header):
            text = content[len(header) :].lstrip("\n")
        else:
            text = content
        retrieved_context.append(
            CheckerContextChunk(
                doc_id=checker_doc_id(source=source, heading_path=heading_path),
                text=text,
            )
        )

    return CheckerSample(
        query_id=artifact["query_id"],
        gold_question=gold["user_question"],
        gt_answer=gold["gt_answer"],
        generator_response=artifact["generator_response"],
        retrieved_context=retrieved_context,
    )
