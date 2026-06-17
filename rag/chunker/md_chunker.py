"""
Markdown-aware token-budget chunker.

Run (from repo root):
  python -m rag.chunker.md_chunker
  python -m rag.chunker.md_chunker --file path/to/doc.md
"""
from dataclasses import dataclass
from typing_extensions import List, Dict, Optional
import os
from ..base import BaseChunker, Chunk

    

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

    # metadata 待定
    def run(self, text:str):
        paragraphs = _split_paragraphs_with_headings(text)
        raw_chunks = _chunk_paragraphs(paragraphs, self.chunk_tokens, self.overlap_tokens)
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


def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """根据标题层次分割段落，保持语义完整性"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0
    
    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            buf.clear()
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })
        buf.clear()
    
    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            # 处理标题行
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip('#'))
            title = raw.lstrip('#').strip()
            
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level-1]
            heading_stack.append(title)
            
            char_pos += len(raw) + 1
            continue
        
        # 段落内容累积
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1
    
    flush_buf(char_pos)
    
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    
    return paragraphs


def _chunk_paragraphs(paragraphs: List, chunk_tokens: int, overlap_tokens: int):
    """基于 Token 数量的智能分块"""
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0

    while i < len(paragraphs):
        p = paragraphs[i] # p：当前段落
        p_tokens = _approx_token_len(p["content"])

        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p) # cur: 当前块
            cur_tokens += p_tokens
            i += 1

        else:
            content = "\n\n".join(x["content"] for x in cur) # 生成 content 段落间用 \n\n 隔开
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None) # next 拿的是生成器第一个产出的值

            chunks.append({ #记住 chunks 的数据结构
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })

            # 构建重叠部分
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = [] # 为下一个 cur 保留的部分
                kept_tokens = 0
                for x in reversed(cur): 
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept)) # 因为 kept 的顺序是反的，所以需要 reversed 一下
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0

    if cur: # 遍历完所有 paragraph 还会剩一个 cur
        content = "\n\n".join(x["content"] for x in cur) # 生成 content 段落间用 \n\n 隔开
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None) # next 拿的是生成器第一个产出的值

        chunks.append({
            "content": content,
            "start": start,
            "end": end,
            "heading_path": heading_path,
        })

    return chunks

import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4、text-embedding-3 等
def _approx_token_len(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(text))


_DEFAULT_FIXTURE = """
# 水果
## 热带水果
### 菠萝
菠萝好吃
## 温带水果
### 苹果
苹果也好吃
"""


def _demo_main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Markdown chunker offline demo.")
    parser.add_argument("--file", help="Markdown file to chunk")
    parser.add_argument("--compare-splits", action="store_true", help="Compare paragraph split variants")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
        chunker = MarkdownChunker()
        chunks = chunker.run(text)
        token_lens = [_approx_token_len(c.content) for c in chunks]
        print(f"MarkdownChunker: {len(chunks)} chunks from {args.file}")
        if token_lens:
            print(f"  token range: {min(token_lens)}..{max(token_lens)}")
        for i, chunk in enumerate(chunks[:8]):
            meta = chunk.metadata or {}
            preview = chunk.content[:60].replace("\n", " ")
            print(
                f"  [{i}] heading={meta.get('heading_path', '-')} | {preview}..."
            )
        return

    if args.compare_splits:
        test_text2 = _DEFAULT_FIXTURE.replace("\n", "\n\n")
        split_text1 = _split_paragraphs_with_headings(_DEFAULT_FIXTURE)
        split_text2 = _split_paragraphs_with_headings(test_text2)
        print(json.dumps(split_text1, indent=2, ensure_ascii=False))
        print("---")
        print(json.dumps(split_text2, indent=2, ensure_ascii=False))
        return

    chunker = MarkdownChunker()
    chunks = chunker.run("# 标题\n\n内容")
    assert isinstance(chunks[0], Chunk)
    print(f"MarkdownChunker smoke: {len(chunks)} chunk(s)")
    print(f"  content={chunks[0].content!r}")


if __name__ == "__main__":
    _demo_main()