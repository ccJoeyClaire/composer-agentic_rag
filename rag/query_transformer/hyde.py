from __future__ import annotations

from typing import Optional

from llm.client import LLMClient

from ..base import BaseQueryTransformer

DEFAULT_SYSTEM_PROMPT = """You write short hypothetical knowledge-base passages that would answer the user's question.
Match the language of the question. Write only the passage (2–4 sentences).
No titles, labels, quotes, or meta commentary."""

DEFAULT_USER_TEMPLATE = "Question:\n{query}\n\nHypothetical passage:"


class HyDETransformer(BaseQueryTransformer):
    """
    HyDE (#15): LLM generates a hypothetical answer; retriever embeds that text
    instead of the raw query (pairs with context-enriched index embeddings).
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
        return doc


async def _demo_main() -> None:
    """Integration smoke: HyDE hypothetical passage for one query.

    Run (from repo root):
      python -m rag.query_transformer.hyde
    """
    import argparse
    import os
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
