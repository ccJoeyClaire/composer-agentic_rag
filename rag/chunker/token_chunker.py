"""Fixed-size token window chunker (no markdown or semantic boundaries)."""

from __future__ import annotations

import tiktoken

from ..base import BaseChunker, Chunk
from ..config import get_rag_config

_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(text))


class TokenChunker(BaseChunker):
    """
    Split text into overlapping token windows for eval control baselines.

    Unlike :class:`MarkdownChunker` or :class:`SemanticChunker`, this ignores
    headings and paragraph structure and only respects ``chunk_tokens`` /
    ``overlap_tokens`` from config.
    """

    def __init__(
        self,
        chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        cfg = get_rag_config().chunker
        self.chunk_tokens = max(
            1, chunk_tokens if chunk_tokens is not None else cfg.chunk_tokens
        )
        overlap = overlap_tokens if overlap_tokens is not None else cfg.overlap_tokens
        self.overlap_tokens = max(0, min(overlap, self.chunk_tokens - 1))

    def run(self, text: str) -> list[Chunk]:
        if not text:
            return []

        tokens = _enc.encode(text)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        start_token = 0
        chunk_index = 0

        while start_token < len(tokens):
            end_token = min(start_token + self.chunk_tokens, len(tokens))
            slice_tokens = tokens[start_token:end_token]
            content = _enc.decode(slice_tokens)
            char_start = len(_enc.decode(tokens[:start_token])) if start_token else 0
            char_end = char_start + len(content)

            chunks.append(
                Chunk(
                    content=content,
                    metadata={
                        "start": char_start,
                        "end": char_end,
                        "chunk_index": chunk_index,
                    },
                )
            )

            if end_token >= len(tokens):
                break

            step = self.chunk_tokens - self.overlap_tokens
            start_token += max(1, step)
            chunk_index += 1

        return chunks
