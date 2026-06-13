"""Tests for unified eval dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.loaders.adapters.native import parse_native_record
from eval.loaders.load import load_dataset_config, load_documents
from eval.paths import REPO_ROOT


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


def test_beir_adapter_subset(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"_id": "1", "title": "T1", "text": "body one"}),
                json.dumps({"_id": "2", "title": "", "text": "body two"}),
            ]
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "beir_ds"
    dataset_root.mkdir()
    (dataset_root / "corpus.jsonl").write_text(corpus.read_text(encoding="utf-8"), encoding="utf-8")
    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "adapter": "beir_corpus",
                "corpus": "corpus.jsonl",
                "subset_n": 1,
            }
        ),
        encoding="utf-8",
    )
    # Point eval.paths.dataset_dir for this test via monkeypatch
    from eval import paths

    original = paths.dataset_dir

    def _dir(name: str) -> Path:
        if name == "beir_tmp":
            return dataset_root
        return original(name)

    paths.dataset_dir = _dir  # type: ignore[method-assign]
    try:
        docs = load_documents("beir_tmp")
    finally:
        paths.dataset_dir = original

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "1"
    assert "T1" in docs[0]["text"]
    assert "body one" in docs[0]["text"]


def test_smoke_dataset_config() -> None:
    cfg = load_dataset_config("smoke")
    assert cfg["adapter"] == "native"
    assert cfg["corpus"] == "manifest.jsonl"
