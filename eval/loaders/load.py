"""Resolve dataset config and load normalized documents for indexing."""

from __future__ import annotations

import json
from pathlib import Path

from eval.loaders.schema import DatasetAdapter, DatasetConfig, EvalDocument

DEFAULT_CORPUS_FILE = "manifest.jsonl"
DEFAULT_ADAPTER: DatasetAdapter = "native"


def dataset_config_path(dataset: str) -> Path:
    from eval.paths import dataset_dir

    return dataset_dir(dataset) / "dataset.json"


def load_dataset_config(dataset: str) -> DatasetConfig:
    """Load ``dataset.json`` or return native defaults when missing."""
    path = dataset_config_path(dataset)
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


def load_documents(dataset: str) -> list[EvalDocument]:
    """Single entry: load all indexable documents for a dataset name."""
    config = load_dataset_config(dataset)
    adapter = config.get("adapter", DEFAULT_ADAPTER)
    if adapter == "native":
        from eval.loaders.adapters.native import load_native_documents

        return load_native_documents(dataset, config)
    if adapter == "beir_corpus":
        from eval.loaders.adapters.beir_corpus import load_beir_corpus_documents

        return load_beir_corpus_documents(dataset, config)
    raise ValueError(f"Unknown dataset adapter {adapter!r} for dataset {dataset!r}")
