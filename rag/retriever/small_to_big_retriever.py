"""
Small-to-Big 检索器：向量搜索 small chunk，query 时物化 parent window 返回。

流程：Query → 向量检索 small hits → 读 anchor_window → 拉取 member → 拼接 parent → LLM

规则：
- 单 hit：直接展开该 hit 的 anchor_window（hit 即 anchor，天然居中）
- 多 hit 且 window 重叠：merge_windows 后物化为一个 parent
- 多 hit 且 window 不重叠：每个 cluster 各返回一个 parent（最多 top_k 个）
"""

from __future__ import annotations

from typing import List, Optional

from ..base import BaseRetriever, BaseVectorStore, Chunk
from ..document_augmentation.parent_builder import (
    ANCHOR_WINDOW_KEY,
    CHUNK_ID_KEY,
    CHUNK_ROLE_KEY,
    MATCHED_CHUNK_IDS_KEY,
    PARENT_CONTENT_KEY,
    PARENT_ID_KEY,
    SMALL_SNIPPET_KEY,
    WINDOW_MEMBER_COUNT_KEY,
    cluster_overlapping_hits,
    get_anchor_window,
    materialize_parent_content,
    merge_windows,
    trim_to_token_budget,
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
        parent_token_budget: 物化后截断到该 token 上限（None 则不截断）。
    """
    if not small_hits:
        return []

    # 按 window 重叠关系聚类；单 hit 自然成 1 个 cluster
    clusters = cluster_overlapping_hits(small_hits)
    # 按 cluster 内最高 score 排序，取 top_k 个 cluster
    clusters.sort(
        key=lambda group: max(h.score for h in group),
        reverse=True,
    )

    parents: List[Chunk] = []

    for group in clusters[:top_k]:
        parent = await _materialize_cluster(
            group,
            store=store,
            parent_token_budget=parent_token_budget,
        )
        if parent is not None:
            parents.append(parent)

    return parents


async def _materialize_cluster(
    hits: List[Chunk],
    *,
    store: Optional[BaseVectorStore],
    parent_token_budget: Optional[int],
) -> Optional[Chunk]:
    """
    将一个 hit cluster 物化为单个 parent chunk。

    cluster 内 1 个 hit → 直接用其 anchor_window；
    多个 hit → merge_windows 合并重叠 window 后再物化。
    """
    windows = []
    for hit in hits:
        window = get_anchor_window(hit.metadata or {})
        if window is not None:
            windows.append(window)

    # 无 anchor_window（旧索引或未启用 small-to-big）→ 原样返回 small
    if not windows:
        best = max(hits, key=lambda h: h.score)
        meta = dict(best.metadata or {})
        return Chunk(
            content=best.content,
            metadata={**meta, CHUNK_ROLE_KEY: "small"},
            score=best.score,
        )

    # 单 window 直接用；多 window 合并（重叠 hit 场景）
    merged = merge_windows(windows) if len(windows) > 1 else windows[0]
    member_ids: List[str] = list(merged.get("member_ids") or [])

    # 拉取 window 内所有 member 的正文
    members = await _resolve_members(member_ids, hits, store=store)
    content = materialize_parent_content(members)
    if parent_token_budget:
        content = trim_to_token_budget(content, parent_token_budget)

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
    parent_id = anchor_id if len(windows) == 1 else f"merged:{anchor_id}"

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
            WINDOW_MEMBER_COUNT_KEY: len(member_ids),
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
        retriever = SmallToBigRetriever(inner, store=store, parent_token_budget=1536)
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

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        if not query.strip():
            return []

        # 多召回一些 small hit，expand 去重合并后仍够 top_k 个 parent
        recall_k = max(top_k, top_k * self.recall_multiplier)
        small_hits = await self.inner.aretrieve(query, top_k=recall_k)
        return await expand_small_hits_to_parents(
            small_hits,
            top_k=top_k,
            store=self.store,
            parent_token_budget=self.parent_token_budget,
        )
