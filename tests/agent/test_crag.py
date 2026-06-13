"""Tests for the isolated CRAG subgraph (private CragState) and parent wrapper."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import AgentConfig, build_ReAct_agent, route_after_crag
from agent.reflection.parsers import extract_rag_tool_results, split_rag_chunks
from agent.state import DEFAULT_RAG_TOOL_NAME, AgentState
from agent.subgraph.CRAG import (
    CragConfig,
    build_crag_node,
    build_crag_subgraph,
    compute_verdict,
    decide_action,
)

pytestmark = pytest.mark.unit


def _rag_state(
    *,
    query: str = "What is RAG?",
    raw: str = "chunk one\n\n---\n\nchunk two",
    tool_name: str = DEFAULT_RAG_TOOL_NAME,
) -> AgentState:
    return {
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool_name, "args": {"query": query}, "id": "tc_rag"},
                ],
            ),
            ToolMessage(content=raw, tool_call_id="tc_rag", id="m_rag"),
        ],
        "metadata": {},
    }


def _crag_input(query: str = "q", passages: list[str] | None = None, **kw) -> dict:
    state = {
        "query": query,
        "passages": passages if passages is not None else ["a", "b"],
        "attempt": 1,
        "max_attempts": 2,
        "methods_tried": [],
        "web_used": False,
    }
    state.update(kw)
    return state


async def _score_all_correct(query: str, passages: list[str]) -> list[dict]:
    return [{"index": i, "label": "correct"} for i in range(len(passages))]


async def _score_mixed_for_trim(query: str, passages: list[str]) -> list[dict]:
    labels = [{"index": 0, "label": "correct"}]
    labels.extend({"index": i, "label": "ambiguous"} for i in range(1, len(passages)))
    return labels


async def _score_all_incorrect(query: str, passages: list[str]) -> list[dict]:
    return [{"index": i, "label": "incorrect"} for i in range(len(passages))]


# -- parsers ----------------------------------------------------------------


def test_split_rag_chunks():
    raw = "alpha\n\n---\n\nbeta"
    assert split_rag_chunks(raw) == ["alpha", "beta"]
    assert split_rag_chunks("") == []


def test_extract_rag_tool_results():
    hits = extract_rag_tool_results(_rag_state()["messages"])
    assert len(hits) == 1
    assert hits[0]["query"] == "What is RAG?"
    assert "chunk one" in hits[0]["raw"]


# -- pure verdict / action logic --------------------------------------------


def test_compute_verdict():
    assert compute_verdict([{"index": 0, "label": "correct"}]) == "correct"
    assert compute_verdict([{"index": 0, "label": "incorrect"}]) == "incorrect"
    assert compute_verdict(
        [{"index": 0, "label": "ambiguous"}, {"index": 1, "label": "ambiguous"}]
    ) == "ambiguous"


def test_decide_action():
    assert decide_action("correct", attempt=1, max_attempts=2, web_enabled=False, web_used=False) == "use"
    assert decide_action("incorrect", attempt=1, max_attempts=2, web_enabled=False, web_used=False) == "requery"
    assert decide_action("incorrect", attempt=2, max_attempts=2, web_enabled=False, web_used=False) == "degrade"
    assert decide_action("incorrect", attempt=2, max_attempts=2, web_enabled=True, web_used=False) == "web_fallback"
    assert decide_action("incorrect", attempt=2, max_attempts=2, web_enabled=True, web_used=True) == "degrade"


# -- subgraph end-to-end (private CragState) --------------------------------


@pytest.mark.asyncio
async def test_subgraph_use_when_all_correct():
    graph = build_crag_subgraph(CragConfig(score_fn=_score_all_correct))
    result = await graph.ainvoke(_crag_input(passages=["a", "b"]))
    assert result["verdict"] == "correct"
    assert result["action"] == "use"
    assert result["final_context"] == "a\n\n---\n\nb"


@pytest.mark.asyncio
async def test_subgraph_trims_to_correct_passages():
    graph = build_crag_subgraph(CragConfig(score_fn=_score_mixed_for_trim))
    result = await graph.ainvoke(_crag_input(passages=["keep me", "drop me"]))
    assert result["action"] == "use"
    assert result["final_context"] == "keep me"


@pytest.mark.asyncio
async def test_subgraph_reretrieves_then_succeeds():
    calls = {"n": 0}

    async def score_fn(query: str, passages: list[str]) -> list[dict]:
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"index": 0, "label": "incorrect"}]
        return [{"index": i, "label": "correct"} for i in range(len(passages))]

    async def retrieve_fn(query: str, options: dict) -> str:
        return "better one\n\n---\n\nbetter two"

    graph = build_crag_subgraph(
        CragConfig(score_fn=score_fn, retrieve_fn=retrieve_fn)
    )
    result = await graph.ainvoke(_crag_input(passages=["bad"]))

    assert result["attempt"] == 2
    assert len(result["methods_tried"]) == 1
    assert result["verdict"] == "correct"
    assert result["final_context"] == "better one\n\n---\n\nbetter two"


@pytest.mark.asyncio
async def test_subgraph_degrades_when_exhausted():
    async def retrieve_fn(query: str, options: dict) -> str:
        return "still bad"

    graph = build_crag_subgraph(
        CragConfig(score_fn=_score_all_incorrect, retrieve_fn=retrieve_fn)
    )
    result = await graph.ainvoke(_crag_input(passages=["bad"]))
    assert result["action"] == "degrade"
    assert result["final_context"] == ""


@pytest.mark.asyncio
async def test_subgraph_web_fallback_when_enabled():
    async def retrieve_fn(query: str, options: dict) -> str:
        return "vector bad"

    async def web_fn(query: str) -> str:
        return "web result one"

    graph = build_crag_subgraph(
        CragConfig(
            score_fn=_score_all_incorrect,
            retrieve_fn=retrieve_fn,
            web_fn=web_fn,
            web_enabled=True,
        )
    )
    result = await graph.ainvoke(_crag_input(passages=["bad"]))
    assert result["web_used"] is True
    assert "web" in result["methods_tried"]
    assert "web result one" in result["final_context"]


# -- parent wrapper node ----------------------------------------------------


@pytest.mark.asyncio
async def test_wrapper_skips_without_rag_tool():
    node = build_crag_node(CragConfig(score_fn=_score_all_correct))
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "math_tool", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="42", tool_call_id="tc1", id="m1"),
        ],
        "metadata": {},
    }
    result = await node(state)
    assert result["metadata"]["crag_verdict"] == "skipped"
    assert result["metadata"]["crag_action"] == "use"


@pytest.mark.asyncio
async def test_wrapper_trims_and_rewrites_tool_message():
    node = build_crag_node(CragConfig(score_fn=_score_mixed_for_trim))
    state = _rag_state(raw="keep me\n\n---\n\ndrop me")
    result = await node(state)

    assert result["metadata"]["crag_action"] == "use"
    assert result["metadata"]["crag_verdict"] == "correct"
    assert result["metadata"]["rag_last_raw"] == "keep me"

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[-1].content == "keep me"


# -- graph wiring -----------------------------------------------------------


def test_route_after_crag_always_returns_llm():
    assert route_after_crag({"messages": [], "metadata": {"crag_action": "requery"}}) == "llm"
    assert route_after_crag({"messages": [], "metadata": {"crag_verdict": "skipped"}}) == "llm"


def test_build_react_agent_with_crag_has_crag_node():
    class _FakeLLM:
        pass

    graph = build_ReAct_agent(
        AgentConfig(llm=_FakeLLM(), enable_crag=True),  # type: ignore[arg-type]
    )
    assert "crag_eval" in graph.get_graph().nodes


def test_build_agent_react_crag_pattern():
    class _FakeLLM:
        pass

    from agent.graph import build_agent

    graph = build_agent(
        AgentConfig(llm=_FakeLLM()),  # type: ignore[arg-type]
        pattern="react_crag",
    )
    assert "crag_eval" in graph.get_graph().nodes


# ================================================================================================================
# PowerShell:
#   pytest -c tests/pytest.ini tests/agent/test_crag.py -v
# ================================================================================================================
