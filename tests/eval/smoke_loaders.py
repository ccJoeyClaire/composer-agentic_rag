"""Smoke-dataset native manifest loader used by pytest only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from _eval_.paths import REPO_ROOT, dataset_dir

DocumentFormat = Literal["markdown", "text", "beir", "inline"]
DatasetAdapter = Literal["native", "beir_corpus"]

DEFAULT_CORPUS_FILE = "manifest.jsonl"
DEFAULT_ADAPTER: DatasetAdapter = "native"


class EvalDocument(TypedDict, total=False):
    doc_id: str
    text: str
    path: NotRequired[str]
    format: NotRequired[DocumentFormat]
    title: NotRequired[str]
    meta: NotRequired[dict[str, object]]


class DatasetConfig(TypedDict, total=False):
    dataset_id: NotRequired[str]
    adapter: DatasetAdapter
    corpus: str
    queries: NotRequired[str]
    qrels: NotRequired[str]
    subset_n: NotRequired[int]
    doc_id_field: NotRequired[str]
    text_field: NotRequired[str]
    title_field: NotRequired[str]


def resolve_repo_path(rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _read_file_text(path: Path, fmt: DocumentFormat) -> str:
    if fmt not in ("markdown", "text"):
        raise ValueError(f"Native adapter cannot read file format {fmt!r}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_native_record(raw: dict[str, object]) -> EvalDocument:
    """Parse one manifest line (canonical or legacy ``source`` key)."""
    doc_id = str(raw.get("doc_id") or raw.get("source") or "").strip()
    if not doc_id:
        raise ValueError(f"manifest record missing doc_id/source: {raw!r}")

    inline_text = raw.get("text")
    if inline_text is not None:
        text = str(inline_text)
        doc: EvalDocument = EvalDocument(doc_id=doc_id, text=text, format="inline")
        title = raw.get("title")
        if title is not None:
            doc["title"] = str(title)
        meta = raw.get("meta")
        if isinstance(meta, dict):
            doc["meta"] = meta
        return doc

    rel_path = raw.get("path")
    if rel_path is None:
        raise ValueError(f"manifest record {doc_id!r} needs text or path")
    path = resolve_repo_path(str(rel_path))
    fmt = str(raw.get("format", "markdown"))
    text = _read_file_text(path, fmt)  # type: ignore[arg-type]
    doc = EvalDocument(
        doc_id=doc_id,
        text=text,
        path=str(rel_path),
        format=fmt,  # type: ignore[typeddict-item]
    )
    title = raw.get("title")
    if title is not None:
        doc["title"] = str(title)
    meta = raw.get("meta")
    if isinstance(meta, dict):
        doc["meta"] = meta
    return doc


def dataset_config_path(name: str) -> Path:
    return dataset_dir(name) / "dataset.json"


def load_dataset_config(name: str) -> DatasetConfig:
    """Load ``dataset.json`` or return native defaults when missing."""
    path = dataset_config_path(name)
    if not path.is_file():
        return DatasetConfig(adapter=DEFAULT_ADAPTER, corpus=DEFAULT_CORPUS_FILE)

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: root must be a JSON object")

    adapter = raw.get("adapter", DEFAULT_ADAPTER)
    corpus = raw.get("corpus", DEFAULT_CORPUS_FILE)
    if not isinstance(adapter, str) or not isinstance(corpus, str):
        raise ValueError(f"{path}: adapter and corpus must be strings")

    config: DatasetConfig = DatasetConfig(adapter=adapter, corpus=corpus)  # type: ignore[typeddict-item]
    if "dataset_id" in raw:
        config["dataset_id"] = str(raw["dataset_id"])
    if "queries" in raw:
        config["queries"] = str(raw["queries"])
    if "qrels" in raw:
        config["qrels"] = str(raw["qrels"])
    if "subset_n" in raw:
        config["subset_n"] = int(raw["subset_n"])
    if "doc_id_field" in raw:
        config["doc_id_field"] = str(raw["doc_id_field"])
    if "text_field" in raw:
        config["text_field"] = str(raw["text_field"])
    if "title_field" in raw:
        config["title_field"] = str(raw["title_field"])
    return config


def load_native_documents(name: str, config: DatasetConfig) -> list[EvalDocument]:
    corpus_file = config.get("corpus", DEFAULT_CORPUS_FILE)
    manifest_path = dataset_dir(name) / corpus_file
    documents: list[EvalDocument] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{manifest_path}: each line must be a JSON object")
        documents.append(parse_native_record(raw))
    return documents


def load_documents(name: str) -> list[EvalDocument]:
    """Load all indexable documents for a smoke/native dataset name."""
    config = load_dataset_config(name)
    adapter = config.get("adapter", DEFAULT_ADAPTER)
    if adapter == "native":
        return load_native_documents(name, config)
    raise ValueError(f"Unknown dataset adapter {adapter!r} for dataset {name!r}")
