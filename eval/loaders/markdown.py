from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from eval.paths import REPO_ROOT, dataset_dir


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    path: Path
    format: str


def load_manifest(dataset: str) -> List[ManifestEntry]:
    manifest_path = dataset_dir(dataset) / "manifest.jsonl"
    entries: List[ManifestEntry] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        entries.append(load_manifest_entry(raw))
    return entries


def load_manifest_entry(raw: dict) -> ManifestEntry:
    rel = raw["path"]
    path = Path(rel)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return ManifestEntry(
        source=raw["source"],
        path=path,
        format=raw.get("format", "markdown"),
    )


def read_document_text(entry: ManifestEntry) -> str:
    if entry.format not in ("markdown", "text"):
        raise ValueError(f"Unsupported format {entry.format!r} for {entry.source}")
    if not entry.path.is_file():
        raise FileNotFoundError(f"Document not found: {entry.path}")
    return entry.path.read_text(encoding="utf-8")
