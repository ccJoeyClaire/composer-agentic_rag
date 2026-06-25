"""Query an indexed Codex collection and dump the retrieval trace as JSON.

Prerequisites: run ``python -m get_start.index_example`` first (same collection).

Run (from repo root):
  python -m get_start.retrieve_example
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from rag.build import build_RAG_retriever
from rag.config import get_profile, get_rag_config
from rag.serialize import RetrieveRunMeta, write_retrieve_traces_json

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILE_ID = "baseline"
_COLLECTION = f"getstart_codex_{_PROFILE_ID}"
_QUERY = "在智能体优先的团队里，人类工程师的主要工作是什么？"
_OUTPUT = Path(__file__).resolve().parent / "runs" / "retrieve.json"


async def main() -> None:
    # 1. 加载 .env（embedding key；若 profile 启用 HyDE / contextual / rerank 还需 LLM key）。
    load_dotenv(_REPO_ROOT / ".env")

    # 2. 读取 arg_config.yaml：profile 开关 + retriever.top_k（最终返回条数）。
    #    _COLLECTION 须与 index_example 一致，且该 collection 已 index 过。
    config = get_rag_config()
    profile = get_profile(config, _PROFILE_ID)
    top_k = config.retriever.top_k

    # 3. 按 profile 组装 RAGRetriever：query transform → retrieve → enrich → rerank。
    #    连接 index_example 写入的同一 Qdrant collection（默认 127.0.0.1:6333）。
    retriever = build_RAG_retriever(
        _COLLECTION,
        use_reranker=profile.use_reranker,
        use_contextual=profile.use_contextual,
        use_hyde=profile.use_hyde,
        use_small_to_big=profile.use_small_to_big,
    )

    # 4. 跑完整检索并保留各阶段快照（HyDE / small hits / recall / rerank / final）。
    #    生产路径用 aquery；此处用 aquery_trace 是为了透明化 pipeline。
    result = await retriever.aquery_trace(_QUERY, top_k=top_k)

    # 5. 写出 pretty JSON（meta + traces，indent=4；stages.* 见 rag/serialize.py）。
    write_retrieve_traces_json(
        _OUTPUT,
        [result],
        meta=RetrieveRunMeta(
            profile_id=_PROFILE_ID,
            collection=_COLLECTION,
            config_path=str(_REPO_ROOT / "arg_config.yaml"),
        ),
        top_k=top_k,
    )

    print(
        f"hits={len(result.chunks)} "
        f"trace={sorted(result.metadata.keys())} -> {_OUTPUT}"
    )

    # 6. 关闭 Qdrant 客户端（retriever 可能包 SmallToBigRetriever → VectorRetriever）。
    chain = retriever.retriever
    store = getattr(chain, "store", None) or getattr(getattr(chain, "inner", None), "store", None)
    if store is not None:
        await store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
