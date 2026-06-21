"""
Small-to-Big 检索器：向量搜索 small chunk，query 时物化 parent window 返回。

流程：Query → 向量检索 small hits → 读 anchor_window → 拉取 member → 拼接 parent → LLM

规则：
- 单 hit：直接展开该 hit 的 anchor_window（hit 即 anchor，天然居中）
- 多 hit 且 window 重叠：聚类为 HitCluster，merged_window 后物化为一个 parent
- 多 hit 且 window 不重叠：每个 cluster 各返回一个 parent（最多 top_k 个）

Run (from repo root):
  python -m rag.retriever.small_to_big_retriever
"""

from __future__ import annotations

from typing import List, Optional

from ..base import BaseRetriever, BaseVectorStore, Chunk
from ..chunker.semantic_chunker import _approx_token_len
from ..document_augmentation.parent_builder import (
    ANCHOR_WINDOW_KEY,
    CHUNK_ID_KEY,
    CHUNK_ROLE_KEY,
    HitCluster,
    MATCHED_CHUNK_IDS_KEY,
    PARENT_CONTENT_KEY,
    PARENT_ID_KEY,
    SMALL_SNIPPET_KEY,
    WINDOW_MEMBER_COUNT_KEY,
    cluster_overlapping_hits,
    materialize_parent_content,
)


async def expand_small_hits_to_parents(
    small_hits: List[Chunk],
    *,
    top_k: int,
    store: Optional[BaseVectorStore] = None,
    parent_token_budget: Optional[int] = None,
) -> List[Chunk]:
    """
    将 small 向量命中映射为物化后的 parent chunk 列表。

    Args:
        small_hits: 内层 retriever 返回的 small chunk 命中（带 score）。
        top_k: 最多返回几个 parent（按 cluster 最高分排序）。
        store: 向量库，用于按 chunk_id 拉取 window 内非 hit 的 member。
        parent_token_budget: 与 top_k 并列约束返回数量（总预算 = budget × top_k）；
            超出时减少 parent 数，单个 parent 始终完整物化 merged window。
    """
    if not small_hits:
        return []

    # 按 window 重叠关系聚类；单 hit 自然成 1 个 cluster
    clusters = cluster_overlapping_hits(small_hits)
    # 按 cluster 内最高 score 排序
    clusters.sort(
        key=lambda cluster: max(h.score for h in cluster.hits),
        reverse=True,
    )

    selected = await _select_clusters_within_budget(
        clusters,
        top_k=top_k,
        store=store,
        parent_token_budget=parent_token_budget,
    )

    parents: List[Chunk] = []
    for cluster in selected:
        parent = await _materialize_cluster(cluster, store=store)
        if parent is not None:
            parents.append(parent)

    return parents


async def _select_clusters_within_budget(
    clusters: List[HitCluster],
    *,
    top_k: int,
    store: Optional[BaseVectorStore],
    parent_token_budget: Optional[int],
) -> List[HitCluster]:
    """
    在 score 排序后选取 cluster：top_k 与总 token 预算并列生效。

    总预算 = parent_token_budget × top_k。超出时减少 parent 数量；单个 parent
    始终完整物化 merged window，不做内容截断。
    """
    if not clusters:
        return []

    if parent_token_budget is None:
        return clusters[:top_k]

    selected: List[HitCluster] = []
    total_tokens = 0
    max_total = parent_token_budget * top_k

    for cluster in clusters:
        if len(selected) >= top_k:
            break
        est = await _estimate_cluster_tokens(cluster, store=store)
        if selected and total_tokens + est > max_total:
            break
        selected.append(cluster)
        total_tokens += est

    return selected


async def _estimate_cluster_tokens(
    cluster: HitCluster,
    *,
    store: Optional[BaseVectorStore],
) -> int:
    """估算 cluster 完整物化后的 token 数。"""
    merged = cluster.merged_window()
    if merged is None:
        return max(
            (_approx_token_len(h.content) for h in cluster.hits),
            default=0,
        )

    member_ids: List[str] = list(merged.get("member_ids") or [])
    members = await _resolve_members(member_ids, cluster.hits, store=store)
    return _approx_token_len(materialize_parent_content(members))


async def _materialize_cluster(
    cluster: HitCluster,
    *,
    store: Optional[BaseVectorStore],
) -> Optional[Chunk]:
    """
    将一个 HitCluster 物化为单个 parent chunk。

    有 window → 用 cluster.merged_window() 展开 member 并拼接；
    无 window（旧索引或未启用 small-to-big）→ 原样返回 small。
    """
    hits = cluster.hits
    merged = cluster.merged_window()

    # 无 anchor_window（旧索引或未启用 small-to-big）→ 原样返回 small
    if merged is None:
        best = max(hits, key=lambda h: h.score)
        meta = dict(best.metadata or {})
        return Chunk(
            content=best.content,
            metadata={**meta, CHUNK_ROLE_KEY: "small"},
            score=best.score,
        )

    member_ids: List[str] = list(merged.get("member_ids") or [])

    members = await _resolve_members(member_ids, hits, store=store)
    content = materialize_parent_content(members)

    if not content.strip():
        return None

    best_score = max(h.score for h in hits)
    matched_ids = [
        (h.metadata or {}).get(CHUNK_ID_KEY)
        for h in hits
        if (h.metadata or {}).get(CHUNK_ID_KEY)
    ]
    snippets = [h.content for h in sorted(hits, key=lambda h: h.score, reverse=True)]

    anchor_id = merged.get("anchor_id") or member_ids[0]
    parent_id = anchor_id if len(cluster.windows) == 1 else f"merged:{anchor_id}"

    base_meta = dict(hits[0].metadata or {})
    return Chunk(
        content=content,
        metadata={
            **base_meta,
            CHUNK_ROLE_KEY: "parent",
            PARENT_ID_KEY: parent_id,
            PARENT_CONTENT_KEY: content,
            SMALL_SNIPPET_KEY: snippets[0] if len(snippets) == 1 else snippets,
            MATCHED_CHUNK_IDS_KEY: matched_ids,
            WINDOW_MEMBER_COUNT_KEY: len(members),
            ANCHOR_WINDOW_KEY: merged,
        },
        score=best_score,
    )


async def _resolve_members(
    member_ids: List[str],
    hits: List[Chunk],
    *,
    store: Optional[BaseVectorStore],
) -> List[Chunk]:
    """
    按 member_ids 顺序解析 window 成员 chunk。

    优先从当前 hits 中取（向量检索已返回的）；缺失的通过 store.aretrieve_by_ids 回查。
    注意：store 为 None 时，window 内非 hit 的 member 无法获取，parent 内容会不完整。
    """
    hit_by_id = {
        (h.metadata or {}).get(CHUNK_ID_KEY): h
        for h in hits
        if (h.metadata or {}).get(CHUNK_ID_KEY)
    }

    missing = [cid for cid in member_ids if cid not in hit_by_id]
    fetched: List[Chunk] = []
    if missing and store is not None:
        fetched = await store.aretrieve_by_ids(missing)

    fetched_by_id = {
        (c.metadata or {}).get(CHUNK_ID_KEY): c
        for c in fetched
        if (c.metadata or {}).get(CHUNK_ID_KEY)
    }

    # 保持 member_ids 的文档顺序
    members: List[Chunk] = []
    for cid in member_ids:
        if cid in hit_by_id:
            members.append(hit_by_id[cid])
        elif cid in fetched_by_id:
            members.append(fetched_by_id[cid])
    return members


class SmallToBigRetriever(BaseRetriever):
    """
    包装稠密检索器：搜索 small chunk，返回物化后的 parent span（#3 Small-to-Big）。

    用法：
        inner = VectorRetriever(embedder=embedder, store=store)
        retriever = SmallToBigRetriever(inner, store=store, parent_token_budget=3072)
        parents = await retriever.aretrieve("query", top_k=5)
    """

    def __init__(
        self,
        inner: BaseRetriever,
        *,
        store: Optional[BaseVectorStore] = None,
        recall_multiplier: int = 4,
        parent_token_budget: Optional[int] = None,
    ):
        self.inner = inner
        # store 用于按 chunk_id 回查 window member；默认从 inner（VectorRetriever）取
        self.store = store or getattr(inner, "store", None)
        self.recall_multiplier = max(1, recall_multiplier)
        self.parent_token_budget = parent_token_budget
        self.last_small_hits: List[Chunk] = []

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        if not query.strip():
            return []

        # 多召回一些 small hit，expand 去重合并后仍够 top_k 个 parent
        recall_k = max(top_k, top_k * self.recall_multiplier)
        small_hits = await self.inner.aretrieve(query, top_k=recall_k)
        self.last_small_hits = list(small_hits)
        return await expand_small_hits_to_parents(
            small_hits,
            top_k=top_k,
            store=self.store,
            parent_token_budget=self.parent_token_budget,
        )


async def _demo_main() -> None:
    """Offline smoke: expand synthetic small hits to parent chunks."""
    hits = [
        Chunk(
            content="Paris is the capital.",
            metadata={
                CHUNK_ID_KEY: "doc::0",
                ANCHOR_WINDOW_KEY: {"anchor_id": "doc::0", "member_ids": ["doc::0", "doc::1"]},
            },
            score=0.9,
        ),
        Chunk(
            content="France is in Europe.",
            metadata={
                CHUNK_ID_KEY: "doc::1",
                ANCHOR_WINDOW_KEY: {"anchor_id": "doc::1", "member_ids": ["doc::0", "doc::1"]},
            },
            score=0.8,
        ),
    ]
    parents = await expand_small_hits_to_parents(hits, top_k=2, store=None)
    print(f"expand_small_hits_to_parents: {len(parents)} parent(s)")
    for i, parent in enumerate(parents):
        preview = parent.content[:120].replace("\n", " ")
        print(f"  [{i}] role={parent.metadata.get(CHUNK_ROLE_KEY)} | {preview}...")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo_main())
