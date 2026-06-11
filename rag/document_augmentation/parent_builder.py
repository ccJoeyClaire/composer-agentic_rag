"""
Small-to-Big (#3): lazy parent windows over indexed small chunks.

索引阶段：为每个 small chunk 分配稳定 chunk_id + anchor_window（只存 member_ids，不物化 parent 正文）。
查询阶段：命中的 chunk 以其自身为 anchor 展开 window，按需拼接 member 内容；多 hit 重叠时 merge。

与旧版区别：
- 旧版 index 时贪心分组并写入 parent_content（边界 chunk 易丢上下文）
- 新版 index 时只建覆盖关系，query 时以 hit 为中心物化 parent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple, cast

from ..base import AnchorWindow, Chunk, ChunkMeta
from ..chunker.semantic_chunker import _approx_token_len

# --- metadata keys（写入 Qdrant payload，query 时读取）---

CHUNK_ID_KEY = "chunk_id"              # 稳定 id，格式 "{source}::{index}"
CHUNK_INDEX_KEY = "chunk_index"        # 在文档内的顺序下标
SECTION_ID_KEY = "section_id"          # source + heading_path，标识同一章节
ANCHOR_WINDOW_KEY = "anchor_window"    # {"anchor_id", "member_ids"}，lazy parent 视图
PARENT_ID_KEY = "parent_id"            # query 时生成，标识物化后的 parent
PARENT_CONTENT_KEY = "parent_content"  # query 时生成，物化后的 parent 正文
CHUNK_ROLE_KEY = "chunk_role"          # "small" | "parent"
SMALL_SNIPPET_KEY = "matched_small_content"  # 触发 parent 展开的 small hit 原文
MATCHED_CHUNK_IDS_KEY = "matched_chunk_ids"  # 触发 parent 展开的 child id 列表
WINDOW_MEMBER_COUNT_KEY = "window_member_count"  # 物化 window 包含的 small chunk 数

# 向后兼容旧字段名
PARENT_SMALL_COUNT_KEY = WINDOW_MEMBER_COUNT_KEY

# ===========================================================================
# 索引阶段：为 chunks 打上 id, 并计算 anchor window 
# ===========================================================================

def _section_ranges(chunks: Sequence[Chunk]) -> List[Tuple[int, int]]: #sequence： 有序的序列如 list, tuple
    """按 heading_path 切分 section，返回每个 section 在 chunks 中的 [start, end) 半开区间。"""
    if not chunks:
        return []
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(chunks)):
        prev_heading = (chunks[i - 1].metadata or {}).get("heading_path")
        curr_heading = (chunks[i].metadata or {}).get("heading_path")
        if curr_heading != prev_heading:
            ranges.append((start, i))
            start = i
    ranges.append((start, len(chunks)))
    return ranges 


def _section_bounds(ranges: Sequence[Tuple[int, int]], index: int) -> Tuple[int, int]:
    """查找 index 所在 section 的 [start, end) 边界。"""
    for start, end in ranges:  # 遍历所有 section 的边界
        if start <= index < end:
            return start, end 
    if ranges: # 遍历完所有 Section，都没有符合 index 的章节, 那么：
        return ranges[0][0], ranges[-1][1] #第一个 section 的 start 和最后一个 section 的 end
    return 0, 0


def _build_anchor_window_indices(
    chunks: Sequence[Chunk],
    anchor_idx: int,
    section_start: int,
    section_end: int,
    *,
    parent_token_budget: int,
) -> List[int]:
    """
    以 anchor_idx 为中心，在同 section 内向左右交替扩展，直到 token 预算用尽。

    扩展策略：左右都能扩时，优先扩距离 anchor 更远的一侧，使 hit 尽量处于 window 中心。
    section 首尾只能单侧扩展（语义边界，不跨 heading_path）。
    """
    anchor = chunks[anchor_idx]
    selected = [anchor_idx]
    total_tokens = _approx_token_len(anchor.content)

    left = anchor_idx - 1
    right = anchor_idx + 1

    while left >= section_start or right < section_end: # 开启循环，直到左扩张碰到 section_start 或右扩张碰到 section_end
        left_ok = left >= section_start # 如果还能 left 返回 True，否则返回 False
        right_ok = right < section_end # 如果还能 right 返回 True，否则返回 False

        left_tokens = (
            _approx_token_len(chunks[left].content) if left_ok else None
        )
        right_tokens = (
            _approx_token_len(chunks[right].content) if right_ok else None
        )

        can_left = ( 
            # 需要 3 个条件同时满足：1. left 不超过边界，2. 左边的 chunk 有内容，tokens 不为空, 3, 加上 left 或 right 不超过 token 预算 
            left_ok
            and left_tokens is not None
            and total_tokens + left_tokens <= parent_token_budget
        ) 
        can_right = (
            right_ok
            and right_tokens is not None
            and total_tokens + right_tokens <= parent_token_budget
        )

        if not can_left and not can_right: 
            break

        # 距 anchor 更近的一侧优先扩，保持 window 居中（left/right/anchor_idx 同为全局下标）
        left_dist = anchor_idx - left if left_ok else 10**9
        right_dist = right - anchor_idx if right_ok else 10**9

        if can_left and can_right:
            if left_dist <= right_dist:
                selected.insert(0, left)
                total_tokens += left_tokens  # type: ignore[operator]
                left -= 1
            else:
                selected.append(right)
                total_tokens += right_tokens  # type: ignore[operator]
                right += 1
        elif can_left:
            selected.insert(0, left)
            total_tokens += left_tokens  # type: ignore[operator]
            left -= 1
        else:
            selected.append(right)
            total_tokens += right_tokens  # type: ignore[operator]
            right += 1

    return selected #定位到 window 的索引位置


def assign_parent_chunks(
    small_chunks: List[Chunk],
    *,
    parent_token_budget: int = 3072,
) -> List[Chunk]:
    """
    为每个 small chunk 分配 chunk_id 和 anchor_window（不写 parent_content）。

    流程：
    1. 遍历赋值 chunk_id / chunk_index / section_id
    2. 在每个 section 内，以当前 chunk 为 anchor 构建 window
    3. 将 window 的 member_ids 写入 metadata.anchor_window

    检索时命中 chunk N，直接读其 anchor_window，以 N 为中心的上下文即 parent 视图。
    """
    if not small_chunks:
        return small_chunks

    default_source = (small_chunks[0].metadata or {}).get("source") or "doc"

    # Pass 1: 稳定 id 与 section 标识
    for i, chunk in enumerate(small_chunks):
        meta = dict(chunk.metadata or {}) # 保留原来的 metadata, 下面都是新增的 key
        source = meta.get("source") or default_source
        meta[CHUNK_ID_KEY] = f"{source}::{i}" # 用于作为其他 chunk 的 parent chunk 的 id
        meta[CHUNK_INDEX_KEY] = i
        heading = meta.get("heading_path")
        meta[SECTION_ID_KEY] = f"{source}::{heading}" if heading else f"{source}::__root__"
        meta[CHUNK_ROLE_KEY] = "small"
        chunk.metadata = meta

    if parent_token_budget < 1:
        return small_chunks

    # Pass 2: 每个 chunk 以自身为 anchor 计算 window
    ranges = _section_ranges(small_chunks)
    for anchor_idx, chunk in enumerate(small_chunks):
        section_start, section_end = _section_bounds(ranges, anchor_idx)
        member_indices = _build_anchor_window_indices(
            small_chunks,
            anchor_idx,
            section_start,
            section_end,
            parent_token_budget=parent_token_budget,
        )
        member_ids = [
            (small_chunks[j].metadata or {})[CHUNK_ID_KEY] for j in member_indices
        ]
        meta = dict(chunk.metadata or {})
        meta[ANCHOR_WINDOW_KEY] = {
            "anchor_id": meta[CHUNK_ID_KEY],
            "member_ids": member_ids,
        }
        chunk.metadata = meta

    return small_chunks

# ==============================================================================
# 查询阶段：以 hit 为中心物化 parent
# ==============================================================================

def get_anchor_window(meta: ChunkMeta) -> Optional[AnchorWindow]:
    """从 chunk metadata 安全读取 anchor_window；格式不合法时返回 None。

    ``meta`` 可能来自 Qdrant payload，运行时不保证符合 ``ChunkMeta``，
    故仍用 isinstance 兜底；类型标注仅供静态检查与编辑器补全。
    """
    window = (meta or {}).get(ANCHOR_WINDOW_KEY)
    if not isinstance(window, dict):
        return None
    member_ids = window.get("member_ids")
    if not member_ids:
        return None
    return cast(AnchorWindow, window)


def windows_overlap(a: dict, b: dict) -> bool:
    """两个 window 的 member_ids 有交集则视为重叠。"""
    a_ids = set(a.get("member_ids") or [])
    b_ids = set(b.get("member_ids") or [])
    return bool(a_ids & b_ids)


def merge_windows(windows: Sequence[dict]) -> dict:
    """
    合并多个重叠 window 为一个，member_ids 按 chunk_index 排序去重。

    用于多 hit 场景：top-k 命中落在同一 section 且 window 重叠时，
    合并后再物化，避免向 LLM 重复投喂重叠段落。
    """
    if not windows:
        return {"anchor_id": "", "member_ids": []}

    all_ids: Set[str] = set()
    for window in windows:
        all_ids.update(window.get("member_ids") or [])

    ordered_ids = sorted(all_ids, key=_chunk_index_from_id)

    # anchor_id 取第一个 window；``merged:`` 前缀由物化层（retriever）统一加
    anchor_id = windows[0].get("anchor_id") or ""

    return {"anchor_id": anchor_id, "member_ids": ordered_ids}


def _chunk_index_from_id(chunk_id: str) -> int:
    """从 "source::42" 解析出文档内顺序下标 42。"""
    if "::" not in chunk_id:
        return 0
    try:
        return int(chunk_id.rsplit("::", 1)[-1])
    except ValueError:
        return 0


@dataclass
class HitCluster:
    """一组 anchor_window 相互（传递）重叠的 hit。

    member_ids 是该 cluster 所有 window 覆盖的 chunk_id 并集，
    用于 O(1) 判断新 hit 是否与本 cluster 重叠。
    """

    hits: List[Chunk] = field(default_factory=list)
    windows: List[AnchorWindow] = field(default_factory=list)
    member_ids: Set[str] = field(default_factory=set)

    def overlaps(self, window: AnchorWindow, ids: Set[str]) -> bool:
        """新 window 与本 cluster 是否有重叠（member 交集或 window 重叠）。"""
        return bool(ids & self.member_ids) or any(
            windows_overlap(window, w) for w in self.windows
        )

    def add_hit(self, hit: Chunk, window: AnchorWindow, ids: Set[str]) -> None:
        self.hits.append(hit)
        self.windows.append(window)
        self.member_ids |= ids

    def absorb(self, other: HitCluster) -> None:
        """把另一个 cluster 整个并入本 cluster（传递闭包合并）。"""
        self.hits.extend(other.hits)
        self.windows.extend(other.windows)
        self.member_ids |= other.member_ids

    def merged_window(self) -> AnchorWindow | None:
        """聚类完成后的合并 window；无 window 时返回 None。"""
        if not self.windows:
            return None
        if len(self.windows) == 1:
            return self.windows[0]
        return merge_windows(self.windows)


def cluster_overlapping_hits(hits: List[Chunk]) -> List[HitCluster]:
    """
    将 small hits 按 anchor_window 重叠关系聚类（传递闭包）。

    例：hit A window [1,2,3]，hit B window [3,4,5] → 同一 cluster。
    无 window 的 hit（旧数据或未索引）单独成 cluster。

    传递闭包：一个新 hit 可能同时与多个已有 cluster 重叠，此时它是连接
    这些 cluster 的“桥”，需把它们全部合并成一个。
    """
    if not hits:
        return []

    clusters: List[HitCluster] = []

    for hit in hits:
        window = get_anchor_window(hit.metadata or {})
        if window is None:
            clusters.append(HitCluster(hits=[hit]))
            continue

        ids = set(window.get("member_ids") or [])
        matching = [c for c in clusters if c.overlaps(window, ids)]

        if not matching:
            clusters.append(HitCluster(hits=[hit], windows=[window], member_ids=set(ids)))
            continue

        primary = matching[0]
        primary.add_hit(hit, window, ids)
        for other in matching[1:]:
            primary.absorb(other)
            clusters.remove(other)

    return clusters


def materialize_parent_content(members: Sequence[Chunk]) -> str:
    """
    按 chunk_index 顺序拼接 member 正文，生成 parent 文本。

    若 member 带有 start/end 字符偏移，则裁剪相邻 chunk 的重叠区域，
    避免 small chunk overlap 导致 parent 内容重复。
    """
    if not members:
        return ""
    if len(members) == 1:
        return members[0].content.strip()

    ordered = sorted(
        members,
        key=lambda c: (c.metadata or {}).get(CHUNK_INDEX_KEY, 0),
    )

    parts: List[str] = []
    last_end: Optional[int] = None

    for chunk in ordered:
        meta = chunk.metadata or {}
        start = meta.get("start")
        end = meta.get("end")
        content = chunk.content

        # 利用字符偏移去掉与上一 chunk 重叠的部分
        if (
            start is not None
            and end is not None
            and last_end is not None
            and start < last_end
        ):
            overlap = last_end - start
            if overlap >= len(content):
                continue
            if overlap > 0:
                content = content[overlap:]

        text = content.strip()
        if text:
            parts.append(text)
        if end is not None:
            last_end = end

    return "\n\n".join(parts)
