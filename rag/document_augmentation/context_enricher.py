"""
Context enrichment family (#4–6): headers, enriched embeddings, doc metadata.

- #5 Contextual chunk headers: breadcrumb-style header from heading_path / source
- #4 Context-enriched retrieval: embed header + body; store raw body in payload
- #6 Document augmentation: doc_title, doc_keywords on each chunk metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

from ..base import BaseContextualEnricher, Chunk


def _doc_title(source: str) -> str:
    if not source:
        return ""
    return Path(source).stem or source


def _collect_doc_keywords(chunks: List[Chunk]) -> List[str]:
    seen: Set[str] = set()
    keywords: List[str] = []
    for chunk in chunks:
        heading = (chunk.metadata or {}).get("heading_path")
        if not heading:
            continue
        for part in heading.split(">"):
            token = part.strip()
            if token and token not in seen:
                seen.add(token)
                keywords.append(token)
    return keywords


def build_contextual_header(metadata: dict, *, source: str = "") -> str:
    """Anthropic-style situating header (#5)."""
    lines: List[str] = []
    doc_title = metadata.get("doc_title") or _doc_title(source)
    src = metadata.get("source") or source

    if doc_title:
        lines.append(f"Document: {doc_title}")
    if src and src != doc_title:
        lines.append(f"Source: {src}")

    heading = metadata.get("heading_path")
    if heading:
        lines.append(f"Section: {heading}")

    keywords = metadata.get("doc_keywords") or []
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")

    return "\n".join(lines)


def build_embed_text(header: str, content: str) -> str:
    """Text fed to the embedder (#4)."""
    header = (header or "").strip()
    content = (content or "").strip()
    if header:
        return f"{header}\n\n{content}"
    return content


class ContextualEnricher(BaseContextualEnricher):
    """
    Index-time: attach headers + embed_text + doc augmentation metadata.
    Query-time: optionally prepend stored header to returned chunks for LLM / reranker.
    """

    def __init__(self, *, prepend_on_retrieve: bool = True):
        self.prepend_on_retrieve = prepend_on_retrieve

    async def aenrich_for_index(
        self, chunks: List[Chunk], *, source: str = ""
    ) -> List[Chunk]:
        if not chunks:
            return []

        doc_title = _doc_title(source)
        doc_keywords = _collect_doc_keywords(chunks)
        enriched: List[Chunk] = []

        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            if source:
                meta.setdefault("source", source)
            if doc_title:
                meta["doc_title"] = doc_title
            if doc_keywords:
                meta["doc_keywords"] = doc_keywords

            header = build_contextual_header(meta, source=source)
            if header:
                meta["contextual_header"] = header
            meta["embed_text"] = build_embed_text(header, chunk.content)

            enriched.append(
                Chunk(content=chunk.content, metadata=meta, score=chunk.score)
            )
        return enriched

    async def aenrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not self.prepend_on_retrieve:
            return list(chunks)

        out: List[Chunk] = []
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            header = meta.get("contextual_header") or build_contextual_header(
                meta, source=meta.get("source", "")
            )
            if header and not chunk.content.startswith(header):
                body = chunk.content
                display = f"{header}\n\n{body}"
            else:
                display = chunk.content
            out.append(
                Chunk(content=display, metadata=meta, score=chunk.score)
            )
        return out
