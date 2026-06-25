"""Batch indexing: one document × multiple RAG profiles → Qdrant collections.

Each profile gets its own collection (``{prefix}_{profile_id}``).
Re-indexing an existing collection is idempotent; existing data is replaced.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from rag.build import build_RAG_indexer
from rag.config import get_profile, get_rag_config

from eval.run_config import EvalRunConfig, index_configs_for_matrix


@dataclass(frozen=True)
class IndexResult:
    """Outcome of one profile index run."""

    config: EvalRunConfig
    ok: bool          # True when averify_index passed
    chunk_count: int


async def _index_one(config: EvalRunConfig, source_path: Path) -> IndexResult:
    """Index a single document for one profile.

    Args:
        config:      Run config supplying profile_id and collection name.
        source_path: Path to the source document to index.

    Returns:
        :class:`IndexResult` with pass/fail and chunk count.
    """
    text = source_path.read_text(encoding="utf-8")
    source = source_path.name
    rag_config = get_rag_config()
    profile = get_profile(rag_config, config.profile_id)

    indexer = build_RAG_indexer(
        config.collection,
        use_token_chunker=profile.use_token_chunker,
        use_contextual=profile.use_contextual,
        use_predict_questions=profile.use_predict_questions,
        use_small_to_big=profile.use_small_to_big,
    )

    ok = await indexer.aindex(text, source=source)

    # Count what was actually stored (scroll to avoid in-memory re-computation).
    chunks = []
    offset: str | None = None
    while True:
        batch, offset = await indexer.store.alist_chunks(
            source=source, limit=256, offset=offset
        )
        chunks.extend(batch)
        if offset is None:
            break

    await indexer.store.aclose()
    return IndexResult(config=config, ok=ok, chunk_count=len(chunks))


async def index_profiles(
    matrix: list[EvalRunConfig],
    source_path: Path,
    *,
    concurrency: int = 1,
) -> list[IndexResult]:
    """Index one document for all unique profiles in ``matrix``.

    Deduplicates: each (doc_slug, profile_id) pair is indexed exactly once.
    Profiles are run sequentially by default (``concurrency=1``) to avoid
    saturating the embedding API; raise if the API can handle parallel calls.

    Args:
        matrix:      Full run matrix; duplicated profiles are deduplicated.
        source_path: Path to the source document.
        concurrency: Max simultaneous index jobs.

    Returns:
        One :class:`IndexResult` per unique profile, in matrix order.
    """
    unique_configs = index_configs_for_matrix(matrix)
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(cfg: EvalRunConfig) -> IndexResult:
        async with semaphore:
            return await _index_one(cfg, source_path)

    return await asyncio.gather(*(_guarded(cfg) for cfg in unique_configs))
