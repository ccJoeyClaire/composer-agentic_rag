"""CRAG isolation scenarios and context recall helpers for pytest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable, NotRequired, TypedDict

from agent.reflection.parsers import split_rag_chunks
from agent.subgraph.CRAG import CragConfig, build_crag_subgraph
from rag.base import Chunk
from tests.eval.gold_recall import GoldRagCase, recall_at_k

CragLabel = TypedDict("CragLabel", {"index": int, "label": str})


class GoldAgentContextCase(GoldRagCase):
    case_id: str
    scenario: str
    pattern: str
    rag_profile: NotRequired[str]
    fixture_noise: NotRequired[str]
    fixture_gold: NotRequired[str]
    crag_labels: NotRequired[list[CragLabel]]


def load_agent_context_cases(path: Path) -> list[GoldAgentContextCase]:
    cases: list[GoldAgentContextCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def load_fixture_text(relative_path: str) -> str:
    from _eval_.paths import REPO_ROOT

    path = REPO_ROOT / relative_path
    return path.read_text(encoding="utf-8").strip()


def text_to_chunks(texts: list[str], *, source: str = "") -> list[Chunk]:
    meta = {"source": source} if source else {}
    return [Chunk(content=text, metadata=dict(meta)) for text in texts if text.strip()]


def recall_on_texts(
    texts: list[str],
    case: GoldRagCase,
    *,
    k: int,
    source: str = "",
) -> float:
    return recall_at_k(text_to_chunks(texts, source=source), case, k=k)


def recall_on_context(raw: str, case: GoldRagCase, *, k: int, source: str = "") -> float:
    return recall_on_texts(split_rag_chunks(raw), case, k=k, source=source)


async def run_crag_trim(
    case: GoldAgentContextCase,
    *,
    noise: str,
    gold: str,
    labels: list[CragLabel],
    score_fn: Callable[[str, list[str]], Awaitable[list[dict]]] | None = None,
) -> str:
    """Run CRAG subgraph on fixture passages with preset relevance labels."""

    async def _default_score(_query: str, _passages: list[str]) -> list[dict]:
        return [dict(item) for item in labels]

    config = CragConfig(score_fn=score_fn or _default_score, max_rag_attempts=2)
    subgraph = build_crag_subgraph(config)
    result = await subgraph.ainvoke(
        {
            "query": case["query"],
            "passages": [noise, gold],
            "attempt": 1,
            "max_attempts": 2,
            "methods_tried": [],
            "web_used": False,
        }
    )
    return str(result.get("final_context", "") or "")
