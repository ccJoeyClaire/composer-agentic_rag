"""Index the Codex demo article into Qdrant and dump stored chunks as JSONL.

Prerequisites: ``.env`` with embedding keys; Qdrant on 127.0.0.1:6333.

Run (from repo root):
  python -m get_start.index_example
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from rag.build import build_RAG_indexer
from rag.config import get_profile, get_rag_config
from rag.base import Chunk
from rag.serialize import IndexRunMeta, write_index_chunks_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTICLE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "工程技术：在智能体优先的世界中利用 Codex.md"
)
_PROFILE_ID = "baseline"
_COLLECTION = f"getstart_codex_{_PROFILE_ID}"
_OUTPUT = Path(__file__).resolve().parent / "runs" / "index.jsonl"


async def main() -> None:
    # 1. 加载 .env（EMBEDDING_API_KEY / LLM_API_KEY 等），供 embedder 与可选 enricher 使用。
    load_dotenv(_REPO_ROOT / ".env")

    # 2. 从 arg_config.yaml 读取 profile 开关（chunker / contextual / s2b 等）。
    #    profile 决定 indexer 走哪条索引管线，须与 retrieve_example 使用同一 _PROFILE_ID。
    profile = get_profile(get_rag_config(), _PROFILE_ID)

    # 3. 按 profile 组装 RAGIndexer：chunk → enrich → embed → 写入 Qdrant collection。
    indexer = build_RAG_indexer(
        _COLLECTION,
        use_token_chunker=profile.use_token_chunker,
        use_contextual=profile.use_contextual,
        use_predict_questions=profile.use_predict_questions,
        use_small_to_big=profile.use_small_to_big,
    )

    # 4. 读取 demo 文章；source 写入 chunk metadata，后续按 source 过滤 / 校验索引用。
    text = _ARTICLE.read_text(encoding="utf-8")
    source = _ARTICLE.name

    # 5. 执行索引管线；ok 表示 averify_index 通过（库中 chunk 数与预期一致）。
    ok = await indexer.aindex(text, source=source)

    # 6. 从 Qdrant scroll 已入库 chunk（非内存重算），用于检查 index 实际落库内容与 metadata。
    chunks: list[Chunk] = []
    offset: str | None = None
    while True:
        batch, offset = await indexer.store.alist_chunks(
            source=source,
            limit=256,
            offset=offset,
        )
        chunks.extend(batch)
        if offset is None:
            break

    # 7. 写出 JSONL：首行 run meta，后续每行一个 chunk（content / embed_text / chunk_id 等）。
    write_index_chunks_jsonl(
        _OUTPUT,
        chunks,
        meta=IndexRunMeta(
            profile_id=_PROFILE_ID,
            collection=_COLLECTION,
            config_path=str(_REPO_ROOT / "arg_config.yaml"),
            source=source,
        ),
    )

    print(f"ok={ok} chunks={len(chunks)} -> {_OUTPUT}")

    # 8. 关闭 Qdrant 客户端连接。
    await indexer.store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
