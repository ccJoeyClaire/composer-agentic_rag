"""BEIR-style corpus.jsonl adapter (``_id`` + ``text`` + optional ``title``)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from eval.loaders.schema import DatasetConfig, EvalDocument


def _field_names(config: DatasetConfig) -> tuple[str, str, str]:
    return (
        config.get("doc_id_field") or "_id",
        config.get("text_field") or "text",
        config.get("title_field") or "title",
    )


def _compose_beir_text(title: str, body: str) -> str:
    title = title.strip()
    body = body.strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def iter_beir_corpus_records(
    corpus_path: Path,
    *,
    subset_n: int | None = None,
    doc_id_field: str = "_id",
    text_field: str = "text",
    title_field: str = "title",
) -> Iterator[EvalDocument]:
    count = 0
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{corpus_path}: each line must be a JSON object")
            doc_id = str(raw.get(doc_id_field, "")).strip()
            body = str(raw.get(text_field, ""))
            if not doc_id:
                raise ValueError(f"{corpus_path}: missing {doc_id_field!r}")
            title = str(raw.get(title_field, "") or "")
            text = _compose_beir_text(title, body)
            meta = raw.get("metadata")
            doc: EvalDocument = EvalDocument(
                doc_id=doc_id,
                text=text,
                format="beir",
            )
            if title:
                doc["title"] = title
            if isinstance(meta, dict):
                doc["meta"] = meta
            yield doc
            count += 1
            if subset_n is not None and count >= subset_n:
                break


def load_beir_corpus_documents(dataset: str, config: DatasetConfig) -> list[EvalDocument]:
    from eval.paths import dataset_dir

    corpus_file = config.get("corpus")
    if not corpus_file:
        raise ValueError("beir_corpus adapter requires corpus path in dataset.json")
    corpus_path = dataset_dir(dataset) / corpus_file
    doc_id_field, text_field, title_field = _field_names(config)
    subset_n = config.get("subset_n")
    return list(
        iter_beir_corpus_records(
            corpus_path,
            subset_n=subset_n,
            doc_id_field=doc_id_field,
            text_field=text_field,
            title_field=title_field,
        )
    )
