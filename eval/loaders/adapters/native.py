"""Native manifest adapter — repo files or inline text in ``manifest.jsonl``."""

from __future__ import annotations

import json
from pathlib import Path

from eval.loaders.schema import DatasetConfig, DocumentFormat, EvalDocument
from eval.paths import REPO_ROOT, dataset_dir


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


def load_native_documents(dataset: str, config: DatasetConfig) -> list[EvalDocument]:
    from eval.paths import dataset_dir

    corpus_file = config.get("corpus", "manifest.jsonl")
    manifest_path = dataset_dir(dataset) / corpus_file
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
