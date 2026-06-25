"""Run agent_v2 with v1-style pattern names.

Prerequisites:
  1. ``.env`` with LLM + embedding keys (CRAG patterns need RERANK_API_KEY)
  2. ``python -m get_start.index_example`` (same collection)

Run (from repo root):

  python -m get_start.agent_example --pattern self_rag
  python -m get_start.agent_example --pattern crag
  python -m get_start.agent_example --pattern crag_self_rag
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent_v2.output import OutputState
from agent_v2.pattern.common import RequestConfig, build_graph
from agent_v2.pattern.config import get_agent_pattern_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILE_ID = "baseline"
COLLECTION = f"getstart_codex_{_PROFILE_ID}"
DEFAULT_QUERY = "在智能体优先的团队里，人类工程师的主要工作是什么？"
RUNS_DIR = Path(__file__).resolve().parent / "runs"

SMOKE_PATTERNS = ("self_rag", "crag", "crag_self_rag")


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


async def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")

    patterns = sorted(get_agent_pattern_config().patterns)
    parser = argparse.ArgumentParser(description="agent_v2 get_start smoke test.")
    parser.add_argument("--pattern", choices=patterns, default="self_rag")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--profile-id", default=_PROFILE_ID)
    parser.add_argument("--web", action="store_true")
    args = parser.parse_args()

    require_env_keys(pattern_id=args.pattern)

    request_config = RequestConfig(
        pattern_id=args.pattern,
        collection=args.collection,
        profile_id=args.profile_id,
        enable_web_search=args.web,
    )
    graph = build_graph(request_config)

    raw = await graph.ainvoke(
        {"messages": [HumanMessage(content=args.query)], "metadata": {}}
    )
    output = OutputState.from_state(
        raw,
        query=args.query,
        request_config=request_config,
    )

    json_path = RUNS_DIR / f"agent_{args.pattern}.json"
    txt_path = RUNS_DIR / f"agent_{args.pattern}.txt"
    output.write_json(json_path)
    output.write_doc(txt_path)

    record = output.to_record_schema()
    print(f"pattern={args.pattern} messages={record['message_count']}")
    print(f"metadata_keys={sorted(record['metadata'])}")
    highlights = record["highlights"]
    if highlights.get("gate_verdict") is not None:
        print(f"gate_verdict={highlights['gate_verdict']}")
    if highlights.get("rag_profile") is not None:
        print(f"rag_profile={highlights['rag_profile']}")
    print(output.to_txt())
    print(f"\n-> {json_path}\n-> {txt_path}")


if __name__ == "__main__":
    asyncio.run(main())



