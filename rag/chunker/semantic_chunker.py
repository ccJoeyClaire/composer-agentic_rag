"""
Semantic-ish chunker (lightweight, no embeddings).

This chunker is designed to satisfy `BaseChunker.run(text) -> List[Chunk]` and
work out-of-the-box without extra ML dependencies.

Approach:
1) Reuse Markdown-aware paragraph splitting (heading hierarchy + blank-line paragraphs)
2) Build chunks under a token budget, but allow *semantic boundaries*:
   - if adjacent paragraphs are dissimilar (Jaccard on normalized word sets),
     prefer to flush the current chunk even if there is remaining token budget.

Notes:
- This is not embedding-based semantic segmentation; it's a pragmatic heuristic.
- For PDFs/web pages, you may want a dedicated loader/normalizer upstream.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

import tiktoken

from ..base import BaseChunker, Chunk
from .md_chunker import _split_paragraphs_with_headings

# cl100k_base: GPT-3.5/4 系列常用的 tokenizer，用于估算 token 数（控制 chunk 大小）。
_enc = tiktoken.get_encoding("cl100k_base")


def _approx_token_len(text: str) -> int:
    """用 tiktoken 编码后的长度近似 token 数，供 chunk 预算与 overlap 计算。"""
    if not text:
        return 0
    return len(_enc.encode(text))


# 从段落文本中提取「词」的正则：
#   - [A-Za-z0-9]+  英文单词、数字
#   - \u4e00-\u9fff 常用汉字（CJK 统一表意文字基本区）
# 标点、空白、符号会被跳过，不参与相似度计算。
_WORD_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _word_set(text: str) -> Set[str]:
    """把文本拆成去重词集合，供 Jaccard 相似度比较相邻段落是否「话题相近」。

    英文转小写以便 "Hello" 与 "hello" 视为同一词；汉字 .lower() 不变。
    """
    return {w.lower() for w in _WORD_RE.findall(text)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard 系数 = |A∩B| / |A∪B|，范围 [0, 1]；越高表示两段共享词越多。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class SemanticChunker(BaseChunker):
    """
    Chunk text into semantically-coherent blocks using lightweight heuristics.

    Args:
        chunk_tokens: hard-ish upper bound for chunk token length.
        overlap_tokens: token budget to overlap between consecutive chunks.
        break_similarity: if similarity between adjacent paragraphs falls below this,
            prefer to break the chunk boundary (when the current chunk is non-empty).
        min_chunk_tokens: do not break *too early*; require current chunk to reach
            at least this many tokens before allowing similarity-based breaks.
    """

    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        *,
        break_similarity: float = 0.18,
        min_chunk_tokens: int = 120,
    ):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.break_similarity = break_similarity
        self.min_chunk_tokens = min_chunk_tokens

    def run(self, text: str) -> List[Chunk]:
        # 先按 Markdown 标题层级 + 空行切成段落（复用 md_chunker 逻辑）。
        paragraphs = _split_paragraphs_with_headings(text)
        raw = _chunk_paragraphs_semantic(
            paragraphs,
            chunk_tokens=self.chunk_tokens,
            overlap_tokens=self.overlap_tokens,
            break_similarity=self.break_similarity,
            min_chunk_tokens=self.min_chunk_tokens,
        )
        return [
            Chunk(
                content=c["content"],
                metadata={
                    k: c[k]
                    for k in ("heading_path", "start", "end", "boundary_reason")
                    if c.get(k) is not None
                },
            )
            for c in raw
        ]


def _chunk_paragraphs_semantic(
    paragraphs: Sequence[Dict],
    *,
    chunk_tokens: int,
    overlap_tokens: int, # 重叠 token
    break_similarity: float, # 相似度阈值
    min_chunk_tokens: int, # 最小 token 下限
) -> List[Dict]:
    """贪心合并段落为 chunk：受 token 上限约束，并在相邻段 Jaccard 过低时主动切分。"""
    chunks: List[Dict] = []
    cur: List[Dict] = []  # 当前正在累积的段落列表
    cur_tokens = 0
    cur_reason: Optional[str] = None

    # 缓存上一段落的词集，避免 flush 后 overlap 时重复计算。
    prev_words: Optional[Set[str]] = None

    def flush(reason: str) -> None:
        """结束当前 chunk，写入 chunks，并按 overlap_tokens 保留尾部段落作为下一块开头。"""
        nonlocal cur, cur_tokens, cur_reason, prev_words
        if not cur:
            return
        content = "\n\n".join(x["content"] for x in cur).strip()
        if not content:
            cur = []
            cur_tokens = 0
            cur_reason = None
            prev_words = None
            return

        start = cur[0].get("start", 0)
        end = cur[-1].get("end", start + len(content))
        heading_path = next(
            (x.get("heading_path") for x in reversed(cur) if x.get("heading_path")),
            None,
        )

        chunks.append(
            {
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
                "boundary_reason": reason,
            }
        )

        # Build overlap: keep as many trailing paragraphs as fit into overlap_tokens.
        if overlap_tokens > 0 and cur:
            kept: List[Dict] = []
            kept_tokens = 0
            for x in reversed(cur):
                t = _approx_token_len(x.get("content", "")) or 1
                if kept_tokens + t > overlap_tokens:
                    break
                kept.append(x)
                kept_tokens += t
            cur = list(reversed(kept))
            cur_tokens = kept_tokens
            prev_words = _word_set(cur[-1]["content"]) if cur else None 
            cur_reason = None
        else:
            cur = []
            cur_tokens = 0
            cur_reason = None
            prev_words = None

    for p in paragraphs: 
        content = (p.get("content") or "").strip()
        if not content:
            continue

        p_tokens = _approx_token_len(content)

        # 语义切分：当前 chunk 已够长时，若新段与上一段词集相似度低于阈值，先 flush。
        if cur and cur_tokens >= min_chunk_tokens:
            cur_last_words = prev_words if prev_words is not None else _word_set(cur[-1]["content"]) #cur_last_words 是段落列表中最后一个段落的词集
            this_words = _word_set(content)
            sim = _jaccard(cur_last_words, this_words)
            if sim < break_similarity:
                flush(reason=f"semantic_break(sim<{break_similarity:.2f})")

        # 硬上限：再加入本段会超出 chunk_tokens 时，先 flush 再接纳新段。
        if cur and (cur_tokens + p_tokens) > chunk_tokens:
            flush(reason="token_limit")

        cur.append(p)
        cur_tokens += p_tokens
        prev_words = _word_set(content)

    if cur:
        flush(reason="end_of_text")

    return chunks

