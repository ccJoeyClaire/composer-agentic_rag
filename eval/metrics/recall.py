from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_gold_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def chunk_matches(case: Dict[str, Any], chunk_text: str, meta: dict) -> bool:
    expected_source = (case.get("expected_source") or "").strip()
    if expected_source:
        source = (meta or {}).get("source") or ""
        if expected_source.lower() not in source.lower():
            return False

    heading_needle = (case.get("expected_heading_contains") or "").strip()
    if heading_needle:
        heading = (meta or {}).get("heading_path") or ""
        if heading_needle.lower() in heading.lower():
            return True

    keywords = case.get("expected_keywords") or []
    haystack = chunk_text.lower()
    meta_text = json.dumps(meta or {}, ensure_ascii=False).lower()
    hits = sum(1 for kw in keywords if kw.lower() in haystack or kw.lower() in meta_text)
    if keywords and hits >= max(1, len(keywords) // 2):
        return True
    return False


def recall_at_k(chunks: Iterable, case: Dict[str, Any], k: int) -> float:
    selected = list(chunks)[:k]
    for chunk in selected:
        text = getattr(chunk, "content", str(chunk))
        meta = getattr(chunk, "metadata", {}) or {}
        if chunk_matches(case, text, meta):
            return 1.0
    return 0.0


def mean_recall_at_k(all_scores: List[float]) -> float:
    if not all_scores:
        return 0.0
    return sum(all_scores) / len(all_scores)
