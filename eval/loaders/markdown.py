"""Backward-compatible shim — prefer ``eval.loaders.load_documents``."""

from __future__ import annotations

from eval.loaders.adapters.native import load_native_documents, parse_native_record, resolve_repo_path
from eval.loaders.load import load_dataset_config, load_documents
from eval.loaders.schema import EvalDocument

# Legacy dataclass name mapped to canonical type for old imports.
ManifestEntry = EvalDocument


def load_manifest(dataset: str) -> list[EvalDocument]:
    return load_documents(dataset)


def load_manifest_entry(raw: dict) -> EvalDocument:
    return parse_native_record(raw)


def read_document_text(entry: EvalDocument) -> str:
    text = entry.get("text")
    if text is None:
        raise ValueError(f"Document {entry.get('doc_id', '?')!r} has no materialized text")
    return text


__all__ = [
    "ManifestEntry",
    "load_manifest",
    "load_manifest_entry",
    "read_document_text",
    "resolve_repo_path",
]
