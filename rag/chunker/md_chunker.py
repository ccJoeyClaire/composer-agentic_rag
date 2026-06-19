"""Markdown-aware token-budget chunker.

Run (from repo root):
  python -m rag.chunker.md_chunker
  python -m rag.chunker.md_chunker --file path/to/doc.md
"""
from __future__ import annotations

from typing import Dict, List

from ..base import BaseChunker, Chunk
from .split_paragraphs import approx_token_len, split_paragraphs


class MarkdownChunker(BaseChunker):
    def __init__(
        self,
        chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        from ..config import get_rag_config

        cfg = get_rag_config().chunker
        self.chunk_tokens = (
            chunk_tokens if chunk_tokens is not None else cfg.chunk_tokens
        )
        self.overlap_tokens = (
            overlap_tokens if overlap_tokens is not None else cfg.overlap_tokens
        )

    def run(self, text: str) -> List[Chunk]:
        paragraphs = split_paragraphs(
            text,
            max_paragraph_tokens=self.chunk_tokens,
        )
        raw_chunks = _chunk_paragraphs(
            paragraphs,
            self.chunk_tokens,
            self.overlap_tokens,
        )
        return [
            Chunk(
                content=c["content"],
                metadata={
                    k: c[k]
                    for k in ("heading_path", "start", "end")
                    if c.get(k) is not None
                },
            )
            for c in raw_chunks
        ]


def _chunk_paragraphs(paragraphs: List, chunk_tokens: int, overlap_tokens: int):
    """Merge pre-split paragraphs into token-budget chunks with overlap."""
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0

    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = approx_token_len(p["content"])

        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1

        else:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next(
                (x["heading_path"] for x in reversed(cur) if x.get("heading_path")),
                None,
            )

            chunks.append(
                {
                    "content": content,
                    "start": start,
                    "end": end,
                    "heading_path": heading_path,
                }
            )

            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0

    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next(
            (x["heading_path"] for x in reversed(cur) if x.get("heading_path")),
            None,
        )

        chunks.append(
            {
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            }
        )

    return chunks


def _demo_main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Markdown chunker offline demo.")
    parser.add_argument("--file", help="Markdown file to chunk")
    args = parser.parse_args()

    if not args.file:
        print("Tip: use `python -m rag.chunker.split_paragraphs` to inspect paragraphs.")
        print("Running chunker on built-in short fixture.")
        text = "# Title\n\nBody paragraph one.\n\nBody paragraph two."
    else:
        text = Path(args.file).read_text(encoding="utf-8")

    chunker = MarkdownChunker()
    chunks = chunker.run(text)
    token_lens = [approx_token_len(c.content) for c in chunks]
    label = args.file or "(fixture)"
    print(f"MarkdownChunker: {len(chunks)} chunks from {label}")
    if token_lens:
        print(f"  token range: {min(token_lens)}..{max(token_lens)}")
    for i, chunk in enumerate(chunks[:12]):
        meta = chunk.metadata or {}
        preview = chunk.content[:60].replace("\n", " ")
        print(f"  [{i}] heading={meta.get('heading_path', '-')} | {preview}...")


if __name__ == "__main__":
    _demo_main()
