"""
Test SemanticChunker on the Codex engineering article, optionally run full RAG.

Article:
  get_start/工程技术：在智能体优先的世界中利用 Codex.md

Usage (from repo root):
  python get_start/rag_demo.py              # chunk inspection only (no API)
  python get_start/rag_demo.py --compare    # semantic vs MarkdownChunker stats
  python get_start/rag_demo.py --rag        # index + query (needs .env + Qdrant)

RAG requires:
  EMBEDDING_API_KEY or LLM_API_KEY
  EMBEDDING_BASE_URL or LLM_BASE_URL
  Qdrant on 127.0.0.1:6333 (docker compose up) unless --in-memory
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

# Windows 控制台：允许 chunk 预览中的中文与特殊标点
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

ARTICLE_PATH = Path(__file__).resolve().parent / "工程技术：在智能体优先的世界中利用 Codex.md"
_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    """用 cl100k_base 编码估算文本 token 数（与 OpenAI 分词器一致）。"""
    return len(_enc.encode(text)) if text else 0


def _safe_print(text: str) -> None:
    """在 GBK 等遗留控制台编码下也能输出 UTF-8 文本。"""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
    sys.stdout.flush()


def _load_article() -> str:
    """读取 demo 用的 Codex 工程文章（Markdown 全文）。"""
    if not ARTICLE_PATH.is_file():
        raise FileNotFoundError(f"Article not found: {ARTICLE_PATH}")
    return ARTICLE_PATH.read_text(encoding="utf-8")


def _print_chunk_report(chunks, *, title: str, preview: int, show_all: bool) -> None:
    """
    打印分块统计：数量、token 分布、边界原因、标题路径，并预览若干 chunk。

    Args:
        chunks: ``Chunk`` 列表（来自 chunker.run）。
        title: 报告标题（如 SemanticChunker / MarkdownChunker）。
        preview: 每条 chunk 正文预览的最大字符数。
        show_all: True 时打印全部 chunk；否则最多 8 条。
    """
    _safe_print(f"\n{'=' * 60}")
    _safe_print(title)
    _safe_print(f"{'=' * 60}")
    _safe_print(f"Chunks: {len(chunks)}")

    if not chunks:
        return

    token_lens = [_token_len(c.content) for c in chunks]
    _safe_print(
        f"Tokens per chunk — min: {min(token_lens)}, "
        f"median: {sorted(token_lens)[len(token_lens) // 2]}, "
        f"max: {max(token_lens)}, total: {sum(token_lens)}"
    )

    reasons = Counter(c.metadata.get("boundary_reason", "—") for c in chunks)
    if reasons:
        _safe_print(f"Boundary reasons: {dict(reasons)}")

    headings = Counter(c.metadata.get("heading_path") or "(no heading)" for c in chunks)
    _safe_print(f"Distinct heading_path: {len(headings)}")

    limit = len(chunks) if show_all else min(8, len(chunks))
    _safe_print(f"\nShowing {limit} / {len(chunks)} chunk(s):\n")

    for i, c in enumerate(chunks[:limit], 1):
        meta = c.metadata or {}
        heading = meta.get("heading_path") or "—"
        reason = meta.get("boundary_reason", "—")
        tokens = _token_len(c.content)
        preview_text = c.content.replace("\n", " ")
        if len(preview_text) > preview:
            preview_text = preview_text[:preview] + "…"

        _safe_print(f"--- [{i}] tokens={tokens} reason={reason} ---")
        _safe_print(f"heading: {heading}")
        _safe_print(preview_text)
        _safe_print("")


def run_chunk_demo(*, compare: bool, preview: int, show_all: bool) -> list:
    """
    仅做分块检查（不调 Embedding API）：对文章跑 SemanticChunker 并打印报告。

    Args:
        compare: 为 True 时额外跑 MarkdownChunker 并对比块数。
        preview: 传给 ``_print_chunk_report`` 的正文预览长度。
        show_all: 是否打印全部分块。

    Returns:
        SemanticChunker 产生的 chunk 列表。
    """
    from rag.chunker.md_chunker import MarkdownChunker
    from rag.chunker.semantic_chunker import SemanticChunker

    text = _load_article()
    _safe_print(f"Article: {ARTICLE_PATH.name}")
    _safe_print(f"Chars: {len(text)}, lines: {text.count(chr(10)) + 1}")

    semantic = SemanticChunker(chunk_tokens=512, overlap_tokens=64)
    sem_chunks = semantic.run(text)
    _print_chunk_report(
        sem_chunks,
        title="SemanticChunker",
        preview=preview,
        show_all=show_all,
    )

    if compare:
        md_chunks = MarkdownChunker(chunk_tokens=512, overlap_tokens=64).run(text)
        _print_chunk_report(
            md_chunks,
            title="MarkdownChunker (token budget only)",
            preview=preview,
            show_all=show_all,
        )
        _safe_print(
            "\nNote: SemanticChunker adds similarity-based breaks "
            f"({len(sem_chunks)} chunks vs {len(md_chunks)} for this article)."
        )

    return sem_chunks


async def run_rag_demo(
    *,
    in_memory: bool,
    top_k: int,
    use_reranker: bool,
    use_contextual: bool,
    use_hyde: bool,
    use_small_to_big: bool,
) -> None:
    """
    端到端 RAG demo：索引 Codex 文章，再对预设中文问题检索并打印命中块。

    不调用 LLM 生成答案，只展示检索结果（chunks + score + metadata）。

    Args:
        in_memory: 传给 ``build_RAG_indexer`` / ``build_RAG_retriever``。
        top_k: 每个问题返回的 chunk 数。
        use_reranker / use_contextual / use_hyde / use_small_to_big: 可选增强开关。
    """
    from rag.build import build_RAG_indexer, build_RAG_retriever

    text = _load_article()
    source = ARTICLE_PATH.name
    tags: list[str] = []
    if use_contextual:
        tags.append("contextual")
    if use_small_to_big:
        tags.append("s2b")
    collection = (
        "codex_semantic_" + "_".join(tags) + "_demo"
        if tags
        else "codex_semantic_demo"
    )

    indexer = build_RAG_indexer(
        collection,
        in_memory=in_memory,
        use_contextual=use_contextual,
        use_small_to_big=use_small_to_big,
    )
    retriever = build_RAG_retriever(
        collection,
        in_memory=in_memory,
        use_reranker=use_reranker,
        use_contextual=use_contextual,
        use_hyde=use_hyde,
        use_small_to_big=use_small_to_big,
        store=indexer.store,
        embedder=indexer.embedder,
    )

    notes = []
    if use_small_to_big:
        notes.append("small-to-big (512 tok child / 3072 parent)")
    if use_contextual:
        notes.append("contextual headers + enriched embed")
    if use_hyde:
        notes.append("HyDE query embed")
    if use_reranker:
        notes.append("CrossEncoder rerank")
    suffix = f" ({', '.join(notes)})" if notes else ""
    _safe_print(
        f"\nIndexing with SemanticChunker → Qdrant (collection={collection}){suffix}…"
    )
    ok = await indexer.aindex(text, source=source)
    _safe_print(f"Index {'succeeded' if ok else 'failed'}\n")

    queries = [
        "在智能体优先的团队里，人类工程师的主要工作是什么？",
        "AGENTS.md 在这个项目里扮演什么角色？",
        "Codex 如何验证 UI 或观测性相关的问题？",
    ]

    for query in queries:
        _safe_print("=" * 60)
        _safe_print(f"Query: {query}\n")
        chunks = await retriever.aquery(query, top_k=top_k)
        if use_hyde and retriever.query_transformer:
            hyde = getattr(retriever.query_transformer, "last_document", None)
            if hyde:
                preview = hyde.replace("\n", " ")
                _safe_print(
                    f"HyDE passage: {preview[:320]}"
                    + ("…" if len(preview) > 320 else "")
                    + "\n"
                )
        if not chunks:
            _safe_print("(no hits — check embedding API / Qdrant)\n")
            continue
        for i, c in enumerate(chunks, 1):
            _safe_print(f"--- [{i}] score={c.score:.4f} ---")
            meta = c.metadata or {}
            role = meta.get("chunk_role", "—")
            heading = meta.get("heading_path") or "—"
            _safe_print(f"role: {role} | heading: {heading}")
            if use_small_to_big and meta.get("window_member_count"):
                _safe_print(
                    f"parent window spans {meta['window_member_count']} small chunk(s)"
                )
            body = c.content.replace("\n", " ")
            _safe_print(body[:500] + ("…" if len(body) > 500 else ""))
            _safe_print("")


def main() -> None:
    """解析 CLI，先跑分块检查；若带 ``--rag`` 再异步执行索引与检索 demo。"""
    parser = argparse.ArgumentParser(description="SemanticChunker + RAG demo on Codex article")
    parser.add_argument(
        "--rag",
        action="store_true",
        help="After chunk report, run embed + index + query (needs API + Qdrant)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run MarkdownChunker and print side-by-side stats",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use Qdrant :memory: (no Docker) when --rag",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Hits per query when --rag")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable Cross-Encoder rerank after vector recall (needs sentence-transformers)",
    )
    parser.add_argument(
        "--contextual",
        action="store_true",
        help="Context enrichment family (#4–6): headers at index, prepend on retrieve",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="HyDE (#15): embed LLM hypothetical passage instead of raw query",
    )
    parser.add_argument(
        "--small-to-big",
        action="store_true",
        help="Small-to-Big (#3): index small chunks, return merged parent spans",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=280,
        help="Max characters per chunk preview",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every chunk in the inspection report",
    )
    args = parser.parse_args()

    run_chunk_demo(compare=args.compare, preview=args.preview, show_all=args.show_all)

    if args.rag:
        asyncio.run(
            run_rag_demo(
                in_memory=args.in_memory,
                top_k=args.top_k,
                use_reranker=args.rerank,
                use_contextual=args.contextual,
                use_hyde=args.hyde,
                use_small_to_big=args.small_to_big,
            )
        )
    else:
        _safe_print("Chunk inspection done. Run with --rag for index + search (needs .env).")


if __name__ == "__main__":
    main()

