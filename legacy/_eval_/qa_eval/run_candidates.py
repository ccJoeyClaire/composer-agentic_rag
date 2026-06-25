"""Run agent on QA gold questions and export candidate answers (scheme B, step 2).

Indexes the source document from ``gold_rubric.jsonl``, runs one or more agent
patterns over every question, and writes ``candidates.jsonl`` for
:mod:`_eval_.qa_eval.score_rubric`.

Example (repo root, requires ``.env`` + Qdrant)::

    python -m _eval_.qa_eval.run_candidates --pattern react --rag-profile baseline

    python -m _eval_.qa_eval.run_candidates \\
        --pattern react --pattern react_crag --rag-profile baseline --recreate

    python -m _eval_.qa_eval.run_candidates --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from tqdm import tqdm

from legacy.agent.graph import AgentConfig, build_agent
from llm.client import LLMClient
from rag.core import RAGIndexer, RAGRetriever
from tools.LocalTool.RAG_tool import bind_retriever
from tools.tool_box import ToolBox

from _eval_.config import DEFAULT_AGENT_RAG_TOP_K, DEFAULT_RECURSION_LIMIT
from _eval_.data_preparing.beir import CorpusDoc
from _eval_.paths import REPO_ROOT
from _eval_.qa_eval.easy_export import load_gold_rubric, slugify_doc_id
from _eval_.qa_eval.score_rubric import CandidateAnswer
from _eval_.qa_eval.types import RubricGold
from _eval_.reflection_eval.beir_runner import build_agent_rag
from _eval_.reflection_eval.shared import collection_exists, drop_collection, final_answer, index_doc_list

_DEFAULT_GOLD = REPO_ROOT / "_eval_" / "datasets" / "Easy-Dataset" / "gold_rubric.jsonl"
_DEFAULT_OUT = REPO_ROOT / "_eval_" / "datasets" / "Easy-Dataset" / "candidates.jsonl"
_DEFAULT_RAG_PROFILE = "baseline"
_DEFAULT_INDEX_CONCURRENCY = 8
_LOCAL_TOOL_PACKAGES = ("tools.LocalTool",)

QA_SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Before answering, call RAG_search_tool "
    "to retrieve evidence from the indexed knowledge base. Base your answer ONLY "
    "on the retrieved passages; if the evidence is insufficient, say so explicitly. "
    "Answer in the same language as the question. End with a concise, direct final answer."
)


@dataclass(frozen=True)
class QaCandidateRunConfig:
    """Inputs for one QA candidate generation run."""

    gold_path: Path
    out_path: Path
    patterns: tuple[str, ...]
    rag_profile: str
    agent_rag_top_k: int
    recursion_limit: int
    index_concurrency: int
    recreate: bool
    limit: int | None
    query_ids: frozenset[str] | None
    append: bool


def qa_collection_name(doc_id: str, rag_profile: str) -> str:
    """Build a Qdrant-safe collection name for one QA corpus + RAG profile."""
    slug = slugify_doc_id(doc_id)
    safe = re.sub(r"[^a-z0-9_-]+", "-", slug.lower())
    safe = re.sub(r"-+", "-", safe).strip("-") or "doc"
    return f"qa_eval_{safe}_{rag_profile}"


def profile_label(pattern: str, rag_profile: str) -> str:
    """Stable profile id for :mod:`_eval_.qa_eval.score_rubric` grouping."""
    return f"{pattern}_{rag_profile}"


def _resolve_source_bundle(records: list[RubricGold]) -> tuple[str, Path]:
    if not records:
        raise ValueError("gold_rubric is empty")
    doc_ids = {record.get("source_doc_id", "") for record in records}
    paths = {record.get("source_path", "") for record in records}
    if len(doc_ids) != 1 or len(paths) != 1:
        raise ValueError("gold_rubric must reference one source_doc_id and one source_path")
    doc_id = next(iter(doc_ids))
    source_path = Path(next(iter(paths)))
    if not source_path.is_file():
        raise FileNotFoundError(f"source document not found: {source_path}")
    return doc_id, source_path


def _select_records(
    records: list[RubricGold],
    *,
    limit: int | None,
    query_ids: frozenset[str] | None,
) -> list[RubricGold]:
    if query_ids is not None:
        selected = [record for record in records if record.get("query_id", "") in query_ids]
        missing = query_ids - {record.get("query_id", "") for record in selected}
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"unknown query_id(s): {missing_list}")
        return selected
    if limit is not None:
        return records[:limit]
    return records


async def _index_source_doc(
    indexer: RAGIndexer,
    *,
    doc_id: str,
    source_path: Path,
    concurrency: int,
) -> None:
    text = source_path.read_text(encoding="utf-8")
    doc = CorpusDoc(doc_id=doc_id, text=text, title=source_path.name)
    await index_doc_list(indexer, [doc], concurrency=concurrency)


async def _answer_one(
    graph,
    *,
    question: str,
    recursion_limit: int,
) -> str:
    state = await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content=QA_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        },
        config={"recursion_limit": recursion_limit},
    )
    messages: list[BaseMessage] = state["messages"]
    return final_answer(messages)


async def _run_pattern(
    pattern: str,
    *,
    retriever: RAGRetriever,
    records: list[RubricGold],
    cfg: QaCandidateRunConfig,
) -> list[CandidateAnswer]:
    bind_retriever(retriever, top_k=cfg.agent_rag_top_k)  # type: ignore[arg-type]
    llm = LLMClient()
    tool_box = ToolBox(packages=_LOCAL_TOOL_PACKAGES)
    graph = build_agent(AgentConfig(llm=llm, tool_box=tool_box), pattern=pattern)
    label = profile_label(pattern, cfg.rag_profile)

    candidates: list[CandidateAnswer] = []
    for record in tqdm(records, desc=label, unit="query", dynamic_ncols=True):
        answer = await _answer_one(
            graph,
            question=record.get("question", ""),
            recursion_limit=cfg.recursion_limit,
        )
        candidates.append(
            CandidateAnswer(
                query_id=record["query_id"],
                profile=label,
                answer=answer,
            )
        )
    return candidates


def write_candidates(path: Path, rows: list[CandidateAnswer], *, append: bool) -> None:
    """Write candidate answers as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.is_file() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "query_id": row.query_id,
                "profile": row.profile,
                "answer": row.answer,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def run_candidates(cfg: QaCandidateRunConfig) -> list[CandidateAnswer]:
    """Index the QA corpus once, run each pattern, and write candidate answers."""
    records = load_gold_rubric(cfg.gold_path)
    doc_id, source_path = _resolve_source_bundle(records)
    selected = _select_records(records, limit=cfg.limit, query_ids=cfg.query_ids)

    collection = qa_collection_name(doc_id, cfg.rag_profile)
    indexer, retriever = build_agent_rag(cfg.rag_profile, collection)
    if cfg.recreate:
        await drop_collection(indexer.store)

    if cfg.recreate or not await collection_exists(indexer.store):
        print(f"indexing {source_path.name} -> {collection} (doc_id={doc_id})")
        await _index_source_doc(
            indexer,
            doc_id=doc_id,
            source_path=source_path,
            concurrency=cfg.index_concurrency,
        )
    else:
        print(f"reusing collection {collection}")

    all_candidates: list[CandidateAnswer] = []
    try:
        for pattern in cfg.patterns:
            print(f"[{profile_label(pattern, cfg.rag_profile)}] {len(selected)} queries")
            rows = await _run_pattern(
                pattern,
                retriever=retriever,
                records=selected,
                cfg=cfg,
            )
            all_candidates.extend(rows)
    finally:
        await indexer.store.aclose()

    write_candidates(cfg.out_path, all_candidates, append=cfg.append)
    return all_candidates


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Run agent on QA gold questions and export candidates.jsonl."
    )
    parser.add_argument("--gold", type=Path, default=_DEFAULT_GOLD)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Agent pattern (repeatable). Default: react.",
    )
    parser.add_argument("--rag-profile", default=_DEFAULT_RAG_PROFILE)
    parser.add_argument("--agent-rag-top-k", type=int, default=DEFAULT_AGENT_RAG_TOP_K)
    parser.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT)
    parser.add_argument(
        "--index-concurrency",
        type=int,
        default=int(os.environ.get("EVAL_INDEX_CONCURRENCY", _DEFAULT_INDEX_CONCURRENCY)),
    )
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the QA index.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Run only these query_id values (repeatable).",
    )
    parser.add_argument("--append", action="store_true", help="Append to --out instead of overwrite.")
    args = parser.parse_args()

    patterns = tuple(args.patterns or ("react",))
    query_ids = frozenset(args.query_ids) if args.query_ids else None
    cfg = QaCandidateRunConfig(
        gold_path=args.gold,
        out_path=args.out,
        patterns=patterns,
        rag_profile=args.rag_profile,
        agent_rag_top_k=args.agent_rag_top_k,
        recursion_limit=args.recursion_limit,
        index_concurrency=args.index_concurrency,
        recreate=args.recreate,
        limit=args.limit,
        query_ids=query_ids,
        append=args.append,
    )

    results = asyncio.run(run_candidates(cfg))
    profiles = sorted({row.profile for row in results})
    print(f"wrote {len(results)} candidates -> {cfg.out_path}")
    print(f"profiles: {', '.join(profiles)}")


if __name__ == "__main__":
    main()
