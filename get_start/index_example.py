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
from rag.config import get_index_profile, get_rag_config
from rag.base import Chunk
from rag.serialize import IndexRunMeta, write_index_chunks_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTICLE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "工程技术：在智能体优先的世界中利用 Codex.md"
)
INDEX_PROFILE_IDS = (
    "token",
    "semantic",
    "baseline",
    "s2b",
    "predict_q",
    "full",
)
_RUNS_DIR = Path(__file__).resolve().parent / "runs/index"


async def index_profile(profile_id: str, *, text: str, source: str) -> None:
    """Index the demo article for one RAG profile and dump chunks as JSONL."""
    profile = get_index_profile(get_rag_config(), profile_id)
    collection = f"getstart_codex_{profile_id}"
    output = _RUNS_DIR / f"index_{profile_id}.jsonl"

    indexer = build_RAG_indexer(
        collection,
        use_token_chunker=profile.use_token_chunker,
        use_contextual=profile.use_contextual,
        use_predict_questions=profile.use_predict_questions,
        use_small_to_big=profile.use_small_to_big,
    )

    ok = await indexer.aindex(text, source=source)

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

    write_index_chunks_jsonl(
        output,
        chunks,
        meta=IndexRunMeta(
            profile_id=profile_id,
            collection=collection,
            config_path=str(_REPO_ROOT / "arg_config.yaml"),
            source=source,
        ),
    )

    print(f"profile={profile_id} ok={ok} chunks={len(chunks)} -> {output}")
    await indexer.store.aclose()


async def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")

    text = _ARTICLE.read_text(encoding="utf-8")
    source = _ARTICLE.name

    for profile_id in INDEX_PROFILE_IDS:
        await index_profile(profile_id, text=text, source=source)


if __name__ == "__main__":
    asyncio.run(main())
