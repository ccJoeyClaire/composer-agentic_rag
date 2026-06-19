"""Adaptive markdown paragraph splitter (headings + blank-line / line / punctuation fallbacks).

Run (from repo root):
  python -m rag.chunker.split_paragraphs
  python -m rag.chunker.split_paragraphs --file tests/test_data/工程技术：在智能体优先的世界中利用\\ Codex.md
  python -m rag.chunker.split_paragraphs --compare-modes --limit 12
"""

from __future__ import annotations

import enum
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

import tiktoken

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SAMPLE = (
    _REPO_ROOT
    / "tests"
    / "test_data"
    / "工程技术：在智能体优先的世界中利用 Codex.md"
)

_enc = tiktoken.get_encoding("cl100k_base")
_PARAGRAPH_CORE_KEYS = frozenset({"content", "start", "end", "heading_path"})
_PUNCT_SPLIT_RE = re.compile(r"(?<=[.;；。！？!?])")


class ParagraphBoundaryMode(enum.Enum):
    """Primary paragraph boundary inferred from document layout."""

    BLANK_LINE = "blank_line"
    SINGLE_LINE = "single_line"


def approx_token_len(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(text))


def detect_paragraph_boundary_mode(text: str) -> ParagraphBoundaryMode:
    """Use single-line boundaries when the document has few ``\\n\\n`` breaks."""
    if "\n\n" not in text:
        return ParagraphBoundaryMode.SINGLE_LINE

    lines = text.splitlines()
    blank_lines = sum(1 for line in lines if not line.strip())
    content_lines = sum(
        1 for line in lines if line.strip() and not line.lstrip().startswith("#")
    )
    if blank_lines == 0:
        return ParagraphBoundaryMode.SINGLE_LINE
    if content_lines > 0 and blank_lines / content_lines < 0.1:
        return ParagraphBoundaryMode.SINGLE_LINE
    return ParagraphBoundaryMode.BLANK_LINE


def split_paragraphs(
    text: str,
    *,
    max_paragraph_tokens: int | None = None,
    boundary_mode: ParagraphBoundaryMode | None = None,
) -> List[Dict]:
    """Split *text* into heading-aware paragraphs within optional token limits."""
    if boundary_mode is None:
        boundary_mode = detect_paragraph_boundary_mode(text)

    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0

    def append_paragraph(content: str, end_pos: int) -> None:
        stripped = content.strip()
        if not stripped:
            return
        paragraphs.append(
            {
                "content": stripped,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(stripped)),
                "end": end_pos,
            }
        )

    def flush_buf(end_pos: int) -> None:
        if not buf:
            return
        append_paragraph("\n".join(buf), end_pos)
        buf.clear()

    for raw in lines:
        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip("#"))
            title = raw.lstrip("#").strip()
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            char_pos += len(raw) + 1
            continue

        if boundary_mode is ParagraphBoundaryMode.BLANK_LINE:
            if raw.strip() == "":
                flush_buf(char_pos)
                buf = []
            else:
                buf.append(raw)
        else:
            if raw.strip() == "":
                char_pos += len(raw) + 1
                continue
            flush_buf(char_pos)
            line_end = char_pos + len(raw)
            append_paragraph(raw, line_end)
            buf = []

        char_pos += len(raw) + 1

    flush_buf(char_pos)

    if not paragraphs:
        stripped = text.strip()
        paragraphs = [
            {"content": stripped, "heading_path": None, "start": 0, "end": len(stripped)}
        ]

    if max_paragraph_tokens is None:
        return paragraphs
    return _refine_oversized_paragraphs(paragraphs, max_paragraph_tokens)


def _paragraph_extra_fields(paragraph: Dict) -> Dict:
    return {k: v for k, v in paragraph.items() if k not in _PARAGRAPH_CORE_KEYS}


def _hard_split_paragraph_by_tokens(paragraph: Dict, max_tokens: int) -> List[Dict]:
    content = paragraph.get("content") or ""
    if not content.strip():
        return []

    base_start = int(paragraph.get("start", 0))
    heading_path = paragraph.get("heading_path")
    extra = _paragraph_extra_fields(paragraph)

    tokens = _enc.encode(content)
    if len(tokens) <= max_tokens:
        return [paragraph]

    result: List[Dict] = []
    start_token = 0
    while start_token < len(tokens):
        end_token = min(start_token + max_tokens, len(tokens))
        slice_tokens = tokens[start_token:end_token]
        piece = _enc.decode(slice_tokens)
        char_start = len(_enc.decode(tokens[:start_token])) if start_token else 0
        char_end = char_start + len(piece)
        result.append(
            {
                "content": piece,
                "heading_path": heading_path,
                "start": base_start + char_start,
                "end": base_start + char_end,
                **extra,
            }
        )
        start_token = end_token
    return result


def _split_paragraph_by_lines(paragraph: Dict, max_tokens: int) -> List[Dict]:
    content = (paragraph.get("content") or "").strip()
    if not content:
        return []
    if approx_token_len(content) <= max_tokens:
        return [paragraph]

    base_start = int(paragraph.get("start", 0))
    heading_path = paragraph.get("heading_path")
    extra = _paragraph_extra_fields(paragraph)

    lines = content.split("\n")
    line_parts: list[tuple[int, int, str]] = []
    offset = 0
    for i, line in enumerate(lines):
        line_start = offset
        line_end = offset + len(line)
        line_parts.append((line_start, line_end, line))
        offset = line_end + (1 if i + 1 < len(lines) else 0)

    result: List[Dict] = []
    buf: list[tuple[int, int, str]] = []

    def emit_buf() -> None:
        nonlocal buf
        if not buf:
            return
        sub_content = "\n".join(part[2] for part in buf).strip()
        if not sub_content:
            buf = []
            return
        sub_para = {
            "content": sub_content,
            "heading_path": heading_path,
            "start": base_start + buf[0][0],
            "end": base_start + buf[-1][1],
            **extra,
        }
        if approx_token_len(sub_content) > max_tokens:
            result.extend(_refine_one_oversized_paragraph(sub_para, max_tokens))
        else:
            result.append(sub_para)
        buf = []

    for part in line_parts:
        line = part[2]
        if not line.strip():
            emit_buf()
            continue

        candidate = "\n".join(p[2] for p in (*buf, part)).strip()
        if buf and approx_token_len(candidate) > max_tokens:
            emit_buf()
            candidate = line

        if approx_token_len(candidate) > max_tokens:
            emit_buf()
            single = {
                "content": line,
                "heading_path": heading_path,
                "start": base_start + part[0],
                "end": base_start + part[1],
                **extra,
            }
            result.extend(_refine_one_oversized_paragraph(single, max_tokens))
            continue

        buf.append(part)

    emit_buf()
    return result if result else _refine_one_oversized_paragraph(paragraph, max_tokens)


def _split_paragraph_by_punctuation(paragraph: Dict, max_tokens: int) -> List[Dict]:
    content = (paragraph.get("content") or "").strip()
    if not content:
        return []
    if approx_token_len(content) <= max_tokens:
        return [paragraph]

    base_start = int(paragraph.get("start", 0))
    heading_path = paragraph.get("heading_path")
    extra = _paragraph_extra_fields(paragraph)

    parts = [part for part in _PUNCT_SPLIT_RE.split(content) if part.strip()]
    if len(parts) <= 1:
        return _hard_split_paragraph_by_tokens(paragraph, max_tokens)

    search_from = 0
    spans: list[tuple[int, int, str]] = []
    for part in parts:
        idx = content.find(part, search_from)
        if idx < 0:
            idx = search_from
        spans.append((idx, idx + len(part), part))
        search_from = idx + len(part)

    result: List[Dict] = []
    buf: list[tuple[int, int, str]] = []

    def emit_buf() -> None:
        nonlocal buf
        if not buf:
            return
        sub_content = "".join(part[2] for part in buf).strip()
        if not sub_content:
            buf = []
            return
        sub_para = {
            "content": sub_content,
            "heading_path": heading_path,
            "start": base_start + buf[0][0],
            "end": base_start + buf[-1][1],
            **extra,
        }
        if approx_token_len(sub_content) > max_tokens:
            result.extend(_hard_split_paragraph_by_tokens(sub_para, max_tokens))
        else:
            result.append(sub_para)
        buf = []

    for span in spans:
        candidate = "".join(p[2] for p in (*buf, span)).strip()
        if buf and approx_token_len(candidate) > max_tokens:
            emit_buf()
            candidate = span[2].strip()

        if approx_token_len(candidate) > max_tokens:
            emit_buf()
            single = {
                "content": span[2].strip(),
                "heading_path": heading_path,
                "start": base_start + span[0],
                "end": base_start + span[1],
                **extra,
            }
            result.extend(_hard_split_paragraph_by_tokens(single, max_tokens))
            continue

        buf.append(span)

    emit_buf()
    return result if result else _hard_split_paragraph_by_tokens(paragraph, max_tokens)


def _refine_one_oversized_paragraph(paragraph: Dict, max_tokens: int) -> List[Dict]:
    content = (paragraph.get("content") or "").strip()
    if not content:
        return []
    if approx_token_len(content) <= max_tokens:
        return [paragraph]

    if "\n" in content:
        return _split_paragraph_by_lines(paragraph, max_tokens)

    parts = [part for part in _PUNCT_SPLIT_RE.split(content) if part.strip()]
    if len(parts) > 1:
        return _split_paragraph_by_punctuation(paragraph, max_tokens)

    return _hard_split_paragraph_by_tokens(paragraph, max_tokens)


def _refine_oversized_paragraphs(
    paragraphs: Sequence[Dict],
    max_tokens: int,
) -> List[Dict]:
    if max_tokens <= 0:
        return list(paragraphs)

    out: List[Dict] = []
    for paragraph in paragraphs:
        content = (paragraph.get("content") or "").strip()
        if not content:
            continue
        if approx_token_len(content) <= max_tokens:
            out.append(paragraph)
        else:
            out.extend(_refine_one_oversized_paragraph(paragraph, max_tokens))
    return out


def _paragraph_stats(paragraphs: Sequence[Dict]) -> dict[str, object]:
    token_lens = [approx_token_len(p.get("content", "")) for p in paragraphs]
    if not token_lens:
        return {"count": 0}

    heading_counts: dict[str, int] = {}
    for paragraph, _tokens in zip(paragraphs, token_lens):
        heading = paragraph.get("heading_path") or "(none)"
        heading_counts[heading] = heading_counts.get(heading, 0) + 1

    return {
        "count": len(paragraphs),
        "token_min": min(token_lens),
        "token_max": max(token_lens),
        "token_mean": round(statistics.mean(token_lens), 1),
        "token_median": round(statistics.median(token_lens), 1),
        "unique_headings": len(heading_counts),
        "top_headings": sorted(
            heading_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5],
    }


def _print_paragraph_report(
    *,
    source: Path,
    text: str,
    paragraphs: Sequence[Dict],
    mode: ParagraphBoundaryMode,
    max_paragraph_tokens: int | None,
    limit: int,
) -> None:
    stats = _paragraph_stats(paragraphs)
    print(f"source: {source}")
    print(f"chars: {len(text)}  lines: {text.count(chr(10)) + 1}")
    print(f"detected mode: {mode.value}")
    if max_paragraph_tokens is not None:
        print(f"max_paragraph_tokens: {max_paragraph_tokens}")
    print(f"paragraphs: {stats['count']}")
    if stats["count"]:
        print(
            "token range: "
            f"{stats['token_min']}..{stats['token_max']} "
            f"(mean={stats['token_mean']}, median={stats['token_median']})"
        )
        print(f"unique headings: {stats['unique_headings']}")
        print("top headings:")
        for heading, count in stats["top_headings"]:
            print(f"  [{count}] {heading}")
        print()
        show = min(limit, len(paragraphs))
        print(f"preview ({show}/{len(paragraphs)}):")
        for i, paragraph in enumerate(paragraphs[:show]):
            tokens = approx_token_len(paragraph["content"])
            preview = paragraph["content"][:72].replace("\n", " ")
            heading = paragraph.get("heading_path") or "-"
            print(f"  [{i:>3}] tokens={tokens:>4} | {heading} | {preview}...")


def _demo_main() -> None:
    import argparse

    from ..config import get_rag_config

    default_tokens = get_rag_config().chunker.chunk_tokens

    parser = argparse.ArgumentParser(description="Paragraph splitter offline demo.")
    parser.add_argument(
        "--file",
        type=Path,
        default=_DEFAULT_SAMPLE,
        help="Markdown file to split (default: Codex engineering article)",
    )
    parser.add_argument(
        "--max-paragraph-tokens",
        type=int,
        default=default_tokens,
        help=f"Refine paragraphs above this token count (default: {default_tokens})",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip oversized-paragraph fallback refinement",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max paragraphs to print in preview",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Show blank_line vs single_line primary split (no refine)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit paragraph list as JSON instead of human report",
    )
    args = parser.parse_args()

    path = args.file
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    text = path.read_text(encoding="utf-8")
    max_tokens = None if args.no_refine else args.max_paragraph_tokens

    if args.compare_modes:
        for mode in ParagraphBoundaryMode:
            primary = split_paragraphs(text, boundary_mode=mode)
            print(f"=== {mode.value}: {len(primary)} paragraph(s) ===")
            for i, paragraph in enumerate(primary[: args.limit]):
                preview = paragraph["content"][:70].replace("\n", " ")
                print(f"  [{i}] {paragraph.get('heading_path', '-')} | {preview}...")
            print()
        return

    mode = detect_paragraph_boundary_mode(text)
    paragraphs = split_paragraphs(
        text,
        max_paragraph_tokens=max_tokens,
        boundary_mode=mode,
    )

    if args.json:
        print(json.dumps(paragraphs, indent=2, ensure_ascii=False))
        return

    _print_paragraph_report(
        source=path,
        text=text,
        paragraphs=paragraphs,
        mode=mode,
        max_paragraph_tokens=max_tokens,
        limit=args.limit,
    )


if __name__ == "__main__":
    _demo_main()
