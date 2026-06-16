"""Tests for BEIR loaders in _eval_.data_preparing.beir."""

from __future__ import annotations

import json
from pathlib import Path

from _eval_.data_preparing.beir import (
    CorpusDoc,
    iter_corpus,
    load_qrels,
    load_queries,
)


def test_iter_corpus_composes_title_and_body(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "1", "title": "T1", "text": "body one"}),
        encoding="utf-8",
    )
    docs = list(iter_corpus(corpus))
    assert len(docs) == 1
    assert docs[0] == CorpusDoc(doc_id="1", text="T1\n\nbody one", title="T1")


def test_iter_corpus_keep_ids_stops_early(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"_id": "1", "title": "", "text": "one"}),
                json.dumps({"_id": "2", "title": "", "text": "two"}),
            ]
        ),
        encoding="utf-8",
    )
    docs = list(iter_corpus(corpus, keep_ids={"2"}))
    assert [d.doc_id for d in docs] == ["2"]


def test_load_queries(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        json.dumps({"_id": "q1", "text": "what is covid?"}),
        encoding="utf-8",
    )
    queries = load_queries(queries_path)
    assert queries["q1"].query_id == "q1"
    assert queries["q1"].text == "what is covid?"


def test_load_qrels_preserves_graded_scores(tmp_path: Path) -> None:
    qrels_path = tmp_path / "qrels.tsv"
    qrels_path.write_text(
        "query-id\tcorpus-id\tscore\n1\td1\t2\n1\td2\t0\n",
        encoding="utf-8",
    )
    qrels = load_qrels(qrels_path)
    assert qrels["1"]["d1"] == 2
    assert qrels["1"]["d2"] == 0
