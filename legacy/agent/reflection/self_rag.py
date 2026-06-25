"""Self-RAG reflection nodes — pre (retrieve hint) and post (grounded check).

These are plain graph nodes composed directly in ``agent/graph.py``; they only
read/write ``AgentState.metadata`` and never need their own subgraph.

Run (from repo root):
  python -m agent.reflection.self_rag
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from legacy.agent.state import (
    DEFAULT_MAX_RAG_ATTEMPTS,
    AgentMetadata,
    AgentState,
    merge_metadata,
)
from llm.client import LLMClient

ClassifyRetrieveFn = Callable[[str], Awaitable[bool]]
CheckGroundedFn = Callable[[str, str, str], Awaitable[bool]]

SELF_RAG_CLASSIFY_PROMPT = """Decide if answering the user requires searching a knowledge base.

User message:
{message}

Return JSON only: {{"need_retrieve": true|false}}
"""

SELF_RAG_GROUNDED_PROMPT = """Decide if the assistant answer is supported by the retrieved context.

Query:
{query}

Retrieved context:
{context}

Assistant answer:
{answer}

Return JSON only: {{"grounded": true|false}}
"""

@dataclass
class SelfRagConfig:
    llm: LLMClient | None = None
    max_rag_attempts: int = DEFAULT_MAX_RAG_ATTEMPTS
    classify_fn: ClassifyRetrieveFn | None = None
    grounded_fn: CheckGroundedFn | None = None


def last_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def last_ai_answer(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message
    return None


async def default_classify_need_retrieve(llm: LLMClient, text: str) -> bool:
    response = await llm.arequest_llm(
        [
            {
                "role": "user",
                "content": SELF_RAG_CLASSIFY_PROMPT.format(message=text),
            }
        ],
        json_output=True,
    )
    try:
        payload = json.loads(response.content or "{}")
    except json.JSONDecodeError:
        payload = {}
    if "need_retrieve" in payload:
        return bool(payload["need_retrieve"])
    return rule_based_need_retrieve(text)


async def default_check_grounded(
    llm: LLMClient,
    query: str,
    context: str,
    answer: str,
) -> bool:
    response = await llm.arequest_llm(
        [
            {
                "role": "user",
                "content": SELF_RAG_GROUNDED_PROMPT.format(
                    query=query,
                    context=context[:4000],
                    answer=answer[:2000],
                ),
            }
        ],
        json_output=True,
    )
    try:
        payload = json.loads(response.content or "{}")
    except json.JSONDecodeError:
        payload = {}
    return bool(payload.get("grounded", False))


async def self_rag_pre_node(
    state: AgentState,
    *,
    llm: LLMClient | None,
    classify_fn: ClassifyRetrieveFn | None,
    max_rag_attempts: int,
) -> dict:
    human = last_human_message(state["messages"])
    if human is None:
        return merge_metadata(
            state,
            {
                "self_rag_need_retrieve": None,
                "self_rag_retry_allowed": True,
                "max_rag_attempts": max_rag_attempts,
            },
        )

    text = str(human.content or "")
    if classify_fn is not None:
        need_retrieve = await classify_fn(text)
    elif llm is not None:
        need_retrieve = await default_classify_need_retrieve(llm, text)
    else:
        need_retrieve = rule_based_need_retrieve(text)

    if need_retrieve:
        hint = "This question may require knowledge-base search before answering."
    else:
        hint = "This question may be answerable without retrieval."

    return merge_metadata(
        state,
        {
            "self_rag_need_retrieve": need_retrieve,
            "self_rag_retry_allowed": True,
            "self_rag_retrieve_hint": hint,
            "max_rag_attempts": max_rag_attempts,
        },
    )


async def self_rag_post_node(
    state: AgentState,
    *,
    llm: LLMClient | None,
    grounded_fn: CheckGroundedFn | None,
    max_rag_attempts: int,
) -> dict:
    meta = dict(state.get("metadata") or {})
    answer_msg = last_ai_answer(state["messages"])
    if answer_msg is None:
        return merge_metadata(state, {"self_rag_grounded": None})

    context = meta.get("rag_last_raw")
    if not context or not str(context).strip():
        return merge_metadata(state, {"self_rag_grounded": None})

    query = str(meta.get("rag_last_query") or "")
    if not query:
        human = last_human_message(state["messages"])
        query = str(human.content or "") if human else ""

    answer = str(answer_msg.content or "")
    if grounded_fn is not None:
        grounded = await grounded_fn(query, str(context), answer)
    elif llm is not None:
        grounded = await default_check_grounded(llm, query, str(context), answer)
    else:
        grounded = _heuristic_grounded(str(context), answer)

    patch: AgentMetadata = {"self_rag_grounded": grounded}
    if (
        not grounded
        and meta.get("self_rag_retry_allowed", True)
        and int(meta.get("rag_attempt", 0)) < max_rag_attempts
    ):
        patch["self_rag_retry_hint"] = (
            "Previous answer may not be supported by retrieved context. "
            "Search again with a more specific query before answering."
        )
        patch["self_rag_retry_allowed"] = False
    return merge_metadata(state, patch)


def _heuristic_grounded(context: str, answer: str) -> bool:
    if not answer.strip():
        return False
    answer_tokens = {token for token in re.findall(r"\w+", answer.lower()) if len(token) > 3}
    context_lower = context.lower()
    if not answer_tokens:
        return True
    overlap = sum(1 for token in answer_tokens if token in context_lower)
    return overlap >= max(1, len(answer_tokens) // 4)

# ======================= fallback ==========================

RETRIEVE_HINT_KEYWORDS = (
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "explain",
    "define",
    "describe",
    "多少",
    "什么",
    "如何",
    "为什么",
    "哪",
    "介绍",
    "解释",
    "是什么",
)
CHITCHAT_PATTERNS = (
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "你好",
    "谢谢",
    "再见",
    "bye",
)

def rule_based_need_retrieve(text: str) -> bool:
    """Heuristic fallback when LLM classification is unavailable.

    Production graphs use :func:`default_classify_need_retrieve`; this runs only
    when ``llm`` and ``classify_fn`` are both absent (e.g. unit tests), or when
    the LLM returns JSON without a ``need_retrieve`` field.

    Args:
        text: Raw user message to classify.

    Returns:
        ``True`` if the message looks like a knowledge-base question;
        ``False`` for chitchat or too-short input.
    """
    stripped = text.strip()
    if len(stripped) < 3:
        return False

    lowered = stripped.lower()
    if (
        any(pattern in lowered for pattern in CHITCHAT_PATTERNS)
        and "?" not in stripped
        and "？" not in stripped
    ):
        return False

    if "?" in stripped or "？" in stripped:
        return True

    if any(keyword in lowered for keyword in RETRIEVE_HINT_KEYWORDS):
        return True

    if re.search(r"\b(w|h)\w+", lowered):
        return True

    return False


# ========================== demo ==========================

async def _demo_main() -> None:
    """Offline smoke: rule-based retrieve classification and heuristic grounding."""
    samples = [
        ("Hello!", False),
        ("What is retrieval-augmented generation?", True),
        ("谢谢", False),
        ("How does HyDE work?", True),
    ]
    print("=== rule_based_need_retrieve ===")
    for text, expected in samples:
        result = rule_based_need_retrieve(text)
        mark = "ok" if result == expected else "!!"
        print(f"  [{mark}] {text!r} -> {result}")

    state: AgentState = {
        "messages": [
            HumanMessage(content="What is Paris?"),
            AIMessage(content="Paris is a city in Germany with no famous landmarks."),
        ],
        "metadata": {
            "rag_last_raw": "Paris is the capital of France.",
            "rag_last_query": "What is Paris?",
            "rag_attempt": 1,
            "self_rag_retry_allowed": True,
        },
    }
    patch = await self_rag_post_node(state, llm=None, grounded_fn=None, max_rag_attempts=2)
    meta = patch.get("metadata") or {}
    print("\n=== self_rag_post_node (heuristic, llm=None) ===")
    print(f"  grounded={meta.get('self_rag_grounded')}")
    retry = meta.get("self_rag_retry_hint", "(none)")
    print(f"  retry_hint={str(retry)[:80]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo_main())
