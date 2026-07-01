"""Run agent with v1-style pattern names.

Prerequisites:
  1. ``.env`` with LLM + embedding keys (CRAG patterns need RERANK_API_KEY)
  2. ``python -m get_start.index_example`` (same collection per index profile)

Run (from repo root):

  python -m get_start.agent_example
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tqdm import tqdm

from agent.output import OutputState
from agent.pattern.common import RequestConfig, build_run
from agent.pattern.config import get_agent_pattern_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(__file__).resolve().parent / "runs" / "agent"

SMOKE_PATTERNS = ("self_rag", "crag", "crag_self_rag")

# preset_id → (index_profile_id, retrieve_profile_id)
PRESETS: dict[str, tuple[str, str]] = {
    "token": ("token", "plain"),
    # "semantic": ("semantic", "plain"),
    # "semantic_rerank": ("semantic", "rerank"),
    "baseline": ("baseline", "rerank_contextual"),
    "baseline_s2b": ("s2b", "rerank_s2b"),
    "baseline_hyde": ("baseline", "rerank_hyde"),
    # "baseline_predict_q": ("predict_q", "rerank_contextual"),
    # "full": ("full", "full"),
}


class SmokeQuery(TypedDict):
    """One smoke-test question keyed for artifact naming."""

    query_id: str
    query: str


QUERIES: tuple[SmokeQuery, ...] = (
    {
        "query_id": "query_001",
        "query": "在智能体优先的团队里，人类工程师的主要工作是什么？",
    },
    {
        "query_id": "query_002",
        "query": "泰坦尼克号的导演还导演过哪些电影？",
    },
)


def collection_for(index_profile_id: str) -> str:
    """Qdrant collection name for a get_start index profile."""
    return f"getstart_codex_{index_profile_id}"


def require_env_keys(*, pattern_id: str) -> None:
    """Fail fast when required API keys are missing."""
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("Missing LLM_API_KEY in .env")

    pattern = get_agent_pattern_config().patterns[pattern_id]
    if pattern.enable_retrieval_gate and not (
        os.environ.get("RERANK_API_KEY") or os.environ.get("EMBEDDING_API_KEY")
    ):
        raise SystemExit(
            "CRAG pattern needs RERANK_API_KEY (or EMBEDDING_API_KEY) for gate scoring."
        )

    if not os.environ.get("EMBEDDING_API_KEY"):
        raise SystemExit(
            "Missing EMBEDDING_API_KEY — run python -m get_start.index_example first."
        )


async def run_query(
    *,
    pattern_id: str,
    preset_id: str,
    index_profile_id: str,
    retrieve_profile_id: str,
    query_id: str,
    query: str,
    pbar: tqdm | None = None,
) -> None:
    """Invoke the agent for one pattern / preset / query and write artifacts."""
    request_config = RequestConfig(
        pattern_id=pattern_id,
        collection=collection_for(index_profile_id),
        index_profile_id=index_profile_id,
        retrieve_profile_id=retrieve_profile_id,
        enable_web_search=True,
    )
    run = build_run(request_config)
    try:
        raw = await run.graph.ainvoke(
            {"messages": [HumanMessage(content=query)], "metadata": {}}
        )
    finally:
        await run.aclose()

    output = OutputState.from_state(
        raw,
        query=query,
        request_config=request_config,
    )

    artifact_stem = f"{pattern_id}/{preset_id}/{query_id}"
    json_path = RUNS_DIR / f"{artifact_stem}.json"
    txt_path = RUNS_DIR / f"{artifact_stem}.txt"
    output.write_json(json_path)
    output.write_doc(txt_path)

    record = output.to_record_schema()
    print(
        f"pattern={pattern_id} preset={preset_id} "
        f"index={index_profile_id} retrieve={retrieve_profile_id} "
        f"query_id={query_id} messages={record['message_count']}"
    )
    print(f"metadata_keys={sorted(record['metadata'])}")
    highlights = record["highlights"]
    if highlights.get("gate_verdict") is not None:
        print(f"gate_verdict={highlights['gate_verdict']}")
    if highlights.get("rag_profile") is not None:
        print(f"rag_profile={highlights['rag_profile']}")
    print(output.to_txt())
    print(f"\n-> {json_path}\n-> {txt_path}")
    if pbar is not None:
        pbar.update(1)


async def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(SMOKE_PATTERNS) * len(PRESETS) * len(QUERIES)
    with tqdm(total=total_runs, desc="agent smoke", unit="run", dynamic_ncols=True) as pbar:
        for pattern_id in SMOKE_PATTERNS:
            require_env_keys(pattern_id=pattern_id)
            for preset_id, (index_profile_id, retrieve_profile_id) in PRESETS.items():
                pbar.set_postfix(pattern=pattern_id, preset=preset_id, refresh=False)
                await asyncio.gather(
                    *(
                        run_query(
                            pattern_id=pattern_id,
                            preset_id=preset_id,
                            index_profile_id=index_profile_id,
                            retrieve_profile_id=retrieve_profile_id,
                            query_id=spec["query_id"],
                            query=spec["query"],
                            pbar=pbar,
                        )
                        for spec in QUERIES
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())
