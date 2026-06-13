"""Canonical eval dataset types — index/search loaders share these shapes."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

DocumentFormat = Literal["markdown", "text", "beir", "inline"]
DatasetAdapter = Literal["native", "beir_corpus"]


class EvalDocument(TypedDict, total=False):
    """One indexable document after adapter normalization.

    Loaders must return records with ``doc_id`` and materialized ``text``.
    ``path`` / ``format`` are optional trace fields from the source manifest.
    """

    doc_id: str
    text: str
    path: NotRequired[str]
    format: NotRequired[DocumentFormat]
    title: NotRequired[str]
    meta: NotRequired[dict[str, object]]


class DatasetConfig(TypedDict, total=False):
    """Per-dataset config at ``eval/<dataset>/dataset.json`` or ``eval/datasets/<name>/dataset.json``."""

    dataset_id: NotRequired[str]
    adapter: DatasetAdapter
    corpus: str
    queries: NotRequired[str]
    qrels: NotRequired[str]
    subset_n: NotRequired[int]
    doc_id_field: NotRequired[str]
    text_field: NotRequired[str]
    title_field: NotRequired[str]
