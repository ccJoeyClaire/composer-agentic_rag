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
