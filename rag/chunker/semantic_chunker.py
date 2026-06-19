"""
Semantic chunker driven by paragraph embedding similarity.

Approach:
1) Reuse Markdown-aware paragraph splitting (heading hierarchy + blank-line paragraphs)
2) Embed each paragraph via :class:`ChunkerEmbeddingClient`
3) Build chunks under a token budget, breaking when adjacent paragraph cosine
   similarity falls below ``break_similarity`` (after ``min_chunk_tokens``).

Run (from repo root):
  python -m rag.chunker.semantic_chunker
  python -m rag.chunker.semantic_chunker --offline
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Sequence

import tiktoken

from ..base import BaseChunker, Chunk
from .embedding_client import ChunkerEmbeddingClient
from .split_paragraphs import split_paragraphs

_enc = tiktoken.get_encoding("cl100k_base")


def _approx_token_len(text: str) -> int:
    """用 tiktoken 编码后的长度近似 token 数，供 chunk 预算与 overlap 计算。"""
    if not text:
        return 0
    return len(_enc.encode(text))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [0, 1] for non-negative-aligned vectors; [-1, 1] generally."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class _ParagraphEmbedder(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


class SemanticChunker(BaseChunker):
    """
    Chunk text into semantically-coherent blocks using paragraph embeddings.

    Args:
        chunk_tokens: hard-ish upper bound for chunk token length.
        overlap_tokens: token budget to overlap between consecutive chunks.
        break_similarity: if cosine similarity between adjacent paragraphs falls
            below this, prefer to break the chunk boundary (when the current
            chunk is non-empty).
        min_chunk_tokens: do not break *too early*; require current chunk to reach
            at least this many tokens before allowing similarity-based breaks.
        embedding_client: optional client override (for tests or custom providers).
    """

    def __init__(
        self,
        chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
        *,
        break_similarity: float | None = None,
        min_chunk_tokens: int | None = None,
        embedding_client: _ParagraphEmbedder | None = None,
    ) -> None:
        from ..config import get_rag_config

        cfg = get_rag_config().chunker
        self.chunk_tokens = (
            chunk_tokens if chunk_tokens is not None else cfg.chunk_tokens
        )
        self.overlap_tokens = (
            overlap_tokens if overlap_tokens is not None else cfg.overlap_tokens
        )
        self.break_similarity = (
            break_similarity if break_similarity is not None else cfg.break_similarity
        )
        self.min_chunk_tokens = (
            min_chunk_tokens if min_chunk_tokens is not None else cfg.min_chunk_tokens
        )
        self._embedding_client: _ParagraphEmbedder = (
            embedding_client if embedding_client is not None else ChunkerEmbeddingClient()
        )

    def run(self, text: str) -> List[Chunk]:
        paragraphs = split_paragraphs(
            text,
            max_paragraph_tokens=self.chunk_tokens,
        )
        active: List[Dict] = []
        texts: List[str] = []
        for p in paragraphs:
            content = (p.get("content") or "").strip()
            if not content:
                continue
            active.append(p)
            texts.append(content)

        embeddings = self._embedding_client.embed_texts(texts)
        if len(embeddings) != len(active):
            raise RuntimeError(
                f"embedding count mismatch: {len(embeddings)} vectors for "
                f"{len(active)} paragraphs"
            )

        for p, emb in zip(active, embeddings):
            p["_embedding"] = emb

        raw = _chunk_paragraphs_semantic(
            active,
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
    overlap_tokens: int,
    break_similarity: float,
    min_chunk_tokens: int,
) -> List[Dict]:
    """贪心合并段落为 chunk：受 token 上限约束，并在相邻段 embedding 相似度过低时切分。"""
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    cur_reason: Optional[str] = None
    prev_embedding: Optional[Sequence[float]] = None

    def flush(reason: str) -> None:
        nonlocal cur, cur_tokens, cur_reason, prev_embedding
        if not cur:
            return
        content = "\n\n".join(x["content"] for x in cur).strip()
        if not content:
            cur = []
            cur_tokens = 0
            cur_reason = None
            prev_embedding = None
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
            prev_embedding = cur[-1].get("_embedding") if cur else None
            cur_reason = None
        else:
            cur = []
            cur_tokens = 0
            cur_reason = None
            prev_embedding = None

    for p in paragraphs:
        content = (p.get("content") or "").strip()
        if not content:
            continue

        p_tokens = _approx_token_len(content)
        embedding = p.get("_embedding")
        if embedding is None:
            raise ValueError("paragraph missing _embedding; embed before chunking")

        if cur and cur_tokens >= min_chunk_tokens:
            cur_last_emb = (
                prev_embedding
                if prev_embedding is not None
                else cur[-1].get("_embedding")
            )
            if cur_last_emb is not None:
                sim = _cosine_similarity(cur_last_emb, embedding)
                if sim < break_similarity:
                    flush(reason=f"semantic_break(sim<{break_similarity:.2f})")

        if cur and (cur_tokens + p_tokens) > chunk_tokens:
            flush(reason="token_limit")

        cur.append(p)
        cur_tokens += p_tokens
        prev_embedding = embedding

    if cur:
        flush(reason="end_of_text")

    return chunks


_SAMPLE_TEXT = (
    "# RAG\n\n"
    "Retrieval-augmented generation combines a retriever with an LLM.\n\n"
    "## Chunking\n\n"
    "Semantic chunking splits text at low-similarity paragraph boundaries.\n\n"
    "## Retrieval\n\n"
    "The retriever fetches relevant passages at query time."
)


class _OfflineEmbedder:
    """Deterministic fake vectors for offline chunk-boundary smoke tests."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i, text in enumerate(texts):
            base = float((hash(text) % 1000) / 1000.0)
            vectors.append([base, float(i % 5) / 5.0, 1.0 - base, 0.5])
        return vectors


def _demo_main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Semantic chunker demo.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use mock embeddings (no API key)",
    )
    args = parser.parse_args()

    if args.offline:
        chunker = SemanticChunker(
            chunk_tokens=80,
            overlap_tokens=10,
            embedding_client=_OfflineEmbedder(),
        )
    else:
        try:
            chunker = SemanticChunker(chunk_tokens=120, overlap_tokens=16)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            print("Tip: pass --offline for API-free smoke test.", file=sys.stderr)
            sys.exit(1)

    chunks = chunker.run(_SAMPLE_TEXT)
    print(f"SemanticChunker: {len(chunks)} chunks (offline={args.offline})")
    for i, chunk in enumerate(chunks):
        reason = (chunk.metadata or {}).get("boundary_reason", "?")
        preview = chunk.content[:70].replace("\n", " ")
        print(f"  [{i}] {reason} | {preview}...")


if __name__ == "__main__":
    _demo_main()
