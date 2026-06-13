"""
Convert foreign corpus formats into canonical native ``manifest.jsonl``.

Usage (repo root):
  python -m eval.tools.convert_dataset --adapter beir_corpus --input eval/datasets/msmarco/msmarco/corpus.jsonl --output eval/datasets/msmarco_subset/manifest.jsonl --subset 500
  python -m eval.tools.convert_dataset --adapter beir_corpus --input path/to/corpus.jsonl --output eval/datasets/smoke/manifest.jsonl --write-config
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.loaders.adapters.beir_corpus import iter_beir_corpus_records
from eval.loaders.schema import DatasetConfig, EvalDocument
from eval.paths import REPO_ROOT, dataset_dir


def _write_native_manifest(path: Path, documents: list[EvalDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in documents:
            record: dict[str, object] = {
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "format": doc.get("format", "inline"),
            }
            if doc.get("title"):
                record["title"] = doc["title"]
            if doc.get("meta"):
                record["meta"] = doc["meta"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_dataset_config(path: Path, *, corpus_file: str) -> None:
    config = DatasetConfig(adapter="native", corpus=corpus_file)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_beir_corpus(
    *,
    input_path: Path,
    output_path: Path,
    subset_n: int | None,
    doc_id_field: str,
    text_field: str,
    title_field: str,
) -> int:
    documents = list(
        iter_beir_corpus_records(
            input_path,
            subset_n=subset_n,
            doc_id_field=doc_id_field,
            text_field=text_field,
            title_field=title_field,
        )
    )
    _write_native_manifest(output_path, documents)
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert corpus to canonical eval manifest.jsonl.")
    parser.add_argument("--adapter", required=True, choices=["beir_corpus"])
    parser.add_argument("--input", required=True, help="Source corpus.jsonl path")
    parser.add_argument("--output", default=None, help="Output manifest.jsonl (default: <dataset>/manifest.jsonl)")
    parser.add_argument("--dataset", default=None, help="Dataset name; writes under eval/datasets/<name>/")
    parser.add_argument("--subset", type=int, default=None, help="Max documents to export")
    parser.add_argument("--doc-id-field", default="_id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--title-field", default="title")
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Also write dataset.json (adapter=native) beside manifest",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
    elif args.dataset:
        output_path = dataset_dir(args.dataset) / "manifest.jsonl"
    else:
        raise SystemExit("Provide --output or --dataset")

    if args.adapter == "beir_corpus":
        count = convert_beir_corpus(
            input_path=input_path,
            output_path=output_path,
            subset_n=args.subset,
            doc_id_field=args.doc_id_field,
            text_field=args.text_field,
            title_field=args.title_field,
        )
    else:
        raise SystemExit(f"Unsupported adapter {args.adapter!r}")

    print(f"Wrote {count} documents → {output_path}")

    if args.write_config:
        config_path = output_path.parent / "dataset.json"
        _write_dataset_config(config_path, corpus_file=output_path.name)
        print(f"Wrote dataset config → {config_path}")


if __name__ == "__main__":
    main()
