"""Shared wiring for replay CLIs: open an existing Qdrant collection + profile retriever."""

from __future__ import annotations

import os

from rag.build import build_RAG_retriever
from rag.core import RAGRetriever
from rag.store.qdrant_store import QdrantVectorStore

from _eval_.config import collection_name
from _eval_.reflection_eval.beir_runner import load_profile_flags


def resolve_collection(
    dataset: str,
    profile_id: str,
    collection: str | None,
) -> str:
    """Return explicit ``collection`` or the default ``pooleval_{dataset}_{profile}`` name."""
    if collection:
        return collection
    return collection_name(dataset, profile_id)


def open_qdrant_store(collection: str) -> QdrantVectorStore:
    """Open a Qdrant client against an existing collection (no indexing).

    Resolution order for the client backend:
    1. ``QDRANT_PATH`` env (embedded local storage, e.g. ``./qdrant_data``).
    2. ``QDRANT_URL`` env (remote or ``:memory:``).
    3. Default localhost ``127.0.0.1:6333`` (Docker compose).
    """
    if path := os.environ.get("QDRANT_PATH"):
        return QdrantVectorStore(collection=collection, path=path)
    if url := os.environ.get("QDRANT_URL"):
        return QdrantVectorStore(collection=collection, url=url)
    return QdrantVectorStore(collection=collection, host="127.0.0.1", port=6333)


def open_retriever(profile_id: str, collection: str) -> tuple[RAGRetriever, QdrantVectorStore]:
    """Build a retriever that reads from an existing collection without re-indexing."""
    flags = load_profile_flags(profile_id)
    store = open_qdrant_store(collection)

    os.environ.setdefault("HYDE_LOG_PROFILE", profile_id)
    os.environ.setdefault("HYDE_LOG_COLLECTION", collection)

    retriever = build_RAG_retriever(
        collection,
        use_reranker=flags["use_reranker"],
        use_contextual=flags["use_contextual"],
        use_hyde=flags["use_hyde"],
        use_small_to_big=flags["use_small_to_big"],
        store=store,
    )
    return retriever, store


_PROFILE_BOOL_FIELDS = (
    "use_token_chunker",
    "use_contextual",
    "use_small_to_big",
    "use_predict_questions",
    "use_hyde",
    "use_reranker",
)


def _qdrant_backend_label() -> str:
    if os.environ.get("QDRANT_PATH"):
        return f"embedded path ({os.environ['QDRANT_PATH']})"
    if os.environ.get("QDRANT_URL"):
        return f"url ({os.environ['QDRANT_URL']})"
    return "localhost:6333 (default Docker)"


def _print_overview() -> None:
    print(
        """rag_factory — wire replay CLIs to an existing eval index (no re-index)

Role
  Shared bootstrap for ``inspect_collection`` and ``query``. Reads
  ``arg_config.yaml`` profile flags, opens the Qdrant collection left behind
  by ``python -m _eval_.rag_eval.run``, and builds a :class:`RAGRetriever`
  that matches that profile's pipeline (HyDE / reranker / small-to-big / …).

When to use
  * You already ran eval and collections like ``pooleval_nfcorpus_baseline`` exist.
  * You want to replay one query or inspect chunks without touching rag_eval/run.py.

Exports
  resolve_collection   ``--collection`` override or ``pooleval_{dataset}_{profile}``
  open_qdrant_store    Qdrant client (QDRANT_PATH → QDRANT_URL → localhost:6333)
  open_retriever       profile flags + existing store → RAGRetriever

Not responsible for
  Running queries, formatting output, or HyDE JSONL (see query.py / rank_report.py).
"""
    )


def main(argv: list[str] | None = None) -> int:
    """Print module overview and show collection / profile wiring for one profile."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Explain rag_factory wiring (no query execution).",
    )
    parser.add_argument("--dataset", default="nfcorpus", help="BEIR dataset id")
    parser.add_argument(
        "--profile",
        default="semantic_rerank",
        help="arg_config.yaml profile id to resolve",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional collection override (else pooleval_{dataset}_{profile})",
    )
    args = parser.parse_args(argv)

    _print_overview()

    collection = resolve_collection(args.dataset, args.profile, args.collection)
    flags = load_profile_flags(args.profile)

    print("=== Demo: resolve_collection ===")
    print(f"  dataset={args.dataset!r} profile={args.profile!r}")
    print(f"  → collection={collection!r}")
    print()

    print("=== Demo: load_profile_flags (from arg_config.yaml) ===")
    enabled = {k: flags[k] for k in _PROFILE_BOOL_FIELDS if flags[k]}
    disabled = [k for k in _PROFILE_BOOL_FIELDS if not flags[k]]
    print(f"  enabled:  {json.dumps(enabled, indent=2)}")
    print(f"  disabled: {disabled}")
    print()

    print("=== Demo: open_qdrant_store backend ===")
    print(f"  would connect via {_qdrant_backend_label()}")
    print("  set QDRANT_PATH=./qdrant_data for embedded storage after local eval")
    print()

    print("=== Next step: run a real replay query ===")
    print(
        f"  python -m _eval_.analysis.replay.query "
        f"--profile {args.profile} --dataset {args.dataset} "
        f"--collection {collection} --query-id <QID> --top-k 80"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
