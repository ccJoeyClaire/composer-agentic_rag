from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

from llm.client import LLMClient

from ..base import BaseQueryTransformer

DEFAULT_SYSTEM_PROMPT = """You write short hypothetical knowledge-base passages that would answer the user's question.
Match the language of the question. Write only the passage (2–4 sentences).
No titles, labels, quotes, or meta commentary."""

DEFAULT_USER_TEMPLATE = "Question:\n{query}\n\nHypothetical passage:"

HYDE_LOG_ENV = "HYDE_LOG"
HYDE_LOG_PATH_ENV = "HYDE_LOG_PATH"
HYDE_LOG_PROFILE_ENV = "HYDE_LOG_PROFILE"
HYDE_LOG_COLLECTION_ENV = "HYDE_LOG_COLLECTION"


class HydeLogEntry(TypedDict):
    timestamp_utc: str
    query: str
    hyde_document: str
    profile: str | None
    collection: str | None


def _default_hyde_log_path() -> Path:
    """Default JSONL path relative to repo root (``_eval_/hyde_log/hyde.jsonl``)."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "_eval_" / "hyde_log" / "hyde.jsonl"


def _hyde_logging_enabled() -> bool:
    if os.environ.get(HYDE_LOG_ENV, "1") == "0":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def _hyde_log_path() -> Path:
    raw = os.environ.get(HYDE_LOG_PATH_ENV)
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else Path.cwd() / path
    return _default_hyde_log_path()


def hyde_log_path() -> Path:
    return _hyde_log_path()


def append_hyde_log_entry(
    *,
    query: str,
    hyde_document: str,
    profile: str | None = None,
    collection: str | None = None,
) -> None:
    """Append one HyDE transform record as JSONL (no-op when logging disabled)."""
    if not _hyde_logging_enabled():
        return

    entry: HydeLogEntry = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "query": query,
        "hyde_document": hyde_document,
        "profile": profile or os.environ.get(HYDE_LOG_PROFILE_ENV),
        "collection": collection or os.environ.get(HYDE_LOG_COLLECTION_ENV),
    }
    path = _hyde_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class HyDETransformer(BaseQueryTransformer):
    """
    HyDE (#15): LLM generates a hypothetical answer; retriever embeds that text
    instead of the raw query (pairs with context-enriched index embeddings).

    Successful transforms append one JSONL line when ``HYDE_LOG`` is not ``0``
    (disabled automatically under pytest). Override path via ``HYDE_LOG_PATH``.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str = DEFAULT_USER_TEMPLATE,
        temperature: float = 0.0,
    ):
        self.llm = llm_client or LLMClient()
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.temperature = temperature
        self.last_document: Optional[str] = None

    async def atransform(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            self.last_document = None
            return query

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(query=query)},
        ]
        response = await self.llm.arequest_llm(
            messages,
            temperature=self.temperature,
        )
        doc = (response.content or "").strip()
        if not doc:
            self.last_document = None
            return query

        self.last_document = doc
        append_hyde_log_entry(query=query, hyde_document=doc)
        return doc


async def _demo_main() -> None:
    """Integration smoke: HyDE hypothetical passage for one query.

    Run (from repo root):
      python -m rag.query_transformer.hyde
    """
    import argparse
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="HyDE query transform demo.")
    parser.add_argument("--query", default="What is retrieval-augmented generation?")
    args = parser.parse_args()

    if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("Missing LLM_API_KEY (or OPENAI_API_KEY).", file=sys.stderr)
        sys.exit(1)

    transformer = HyDETransformer()
    hyde_text = await transformer.atransform(args.query)
    print(f"Query: {args.query}")
    print(f"HyDE document:\n{hyde_text}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo_main())
