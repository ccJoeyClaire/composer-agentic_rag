"""Tests for smoke native manifest loading."""

from __future__ import annotations

from tests.eval.smoke_loaders import (
    load_dataset_config,
    load_documents,
    parse_native_record,
)


def test_smoke_native_loads_codex_document() -> None:
    docs = load_documents("smoke")
    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"] == "codex.md"
    assert "智能体" in doc["text"]
    assert doc.get("format") == "markdown"


def test_legacy_source_key_still_parses() -> None:
    raw = {
        "source": "legacy.md",
        "path": "get_start/工程技术：在智能体优先的世界中利用 Codex.md",
        "format": "markdown",
    }
    doc = parse_native_record(raw)
    assert doc["doc_id"] == "legacy.md"
    assert "智能体" in doc["text"]


def test_inline_text_record() -> None:
    doc = parse_native_record({"doc_id": "inline-1", "text": "hello corpus"})
    assert doc["text"] == "hello corpus"
    assert doc["format"] == "inline"


def test_smoke_dataset_config() -> None:
    cfg = load_dataset_config("smoke")
    assert cfg["adapter"] == "native"
    assert cfg["corpus"] == "manifest.jsonl"
