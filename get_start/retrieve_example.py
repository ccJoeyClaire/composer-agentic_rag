"""Query indexed Codex collections and dump retrieval traces as JSON.

Prerequisites: run ``python -m get_start.index_example`` first (same collections).

Run (from repo root):
  python -m get_start.retrieve_example
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from rag.build import build_RAG_retriever
from rag.config import get_rag_config, get_retrieve_profile
from rag.serialize import RetrieveRunMeta, write_retrieve_traces_json

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PROFILE_IDS = (
    "token",
    "semantic",
    "baseline",
    "s2b",
    "predict_q",
    "full",
)
_RETRIEVE_PROFILE_IDS = (
    "plain",
    "rerank",
    "rerank_contextual",
    "rerank_s2b",
    "rerank_hyde",
    "full",
)
_QUERY = "在智能体优先的团队里，人类工程师的主要工作是什么？"
_RUNS_DIR = Path(__file__).resolve().parent / "runs" / "retrieve"


async def main(index_profile_id: str, retrieve_profile_id: str) -> None:
    """Run retrieval for one index + retrieve profile pair."""
    collection = f"getstart_codex_{index_profile_id}"
    output = _RUNS_DIR / f"index_profile_{index_profile_id}/{retrieve_profile_id}.json"

    config = get_rag_config()
    retrieve_profile = get_retrieve_profile(config, retrieve_profile_id)
    top_k = config.retriever.top_k

    retriever = build_RAG_retriever(
        collection,
        use_reranker=retrieve_profile.use_reranker,
        use_contextual=retrieve_profile.use_contextual,
        use_hyde=retrieve_profile.use_hyde,
        use_small_to_big=retrieve_profile.use_small_to_big,
    )

    result = await retriever.aquery_trace(_QUERY, top_k=top_k)

    write_retrieve_traces_json(
        output,
        [result],
        meta=RetrieveRunMeta(
            profile_id=f"{index_profile_id}+{retrieve_profile_id}",
            collection=collection,
            config_path=str(_REPO_ROOT / "arg_config.yaml"),
        ),
        top_k=top_k,
    )

    print(
        f"profile={index_profile_id}+{retrieve_profile_id} "
        f"hits={len(result.chunks)} "
        f"trace={sorted(result.metadata.keys())} -> {output}"
    )

    chain = retriever.retriever
    store = getattr(chain, "store", None) or getattr(
        getattr(chain, "inner", None), "store", None
    )
    if store is not None:
        await store.aclose()


if __name__ == "__main__":
    load_dotenv(_REPO_ROOT / ".env")

    async def run_all() -> None:
        for index_profile_id in _INDEX_PROFILE_IDS:
            for retrieve_profile_id in _RETRIEVE_PROFILE_IDS:
                await main(index_profile_id, retrieve_profile_id)

    asyncio.run(run_all())
