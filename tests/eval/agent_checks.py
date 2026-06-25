"""Agent graph behavior checks (pytest-only; not quality eval)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from unittest.mock import AsyncMock, MagicMock

from legacy.agent.graph import AgentConfig, build_agent
from legacy.agent.reflection.feedback import detect_feedback_node, route_after_detect
from legacy.agent.reflection.self_rag import self_rag_pre_node
from legacy.agent.state import AgentState
from legacy.agent.subgraph.CRAG import compute_verdict, decide_action


def _mock_llm_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps(payload)
    return response


async def run_agent_case(case: dict[str, Any]) -> dict[str, Any]:
    check = case["check"]
    if check == "graph_nodes":
        return _check_graph_nodes(case)
    if check == "self_rag_pre":
        return await _check_self_rag_pre(case)
    if check == "crag_decide":
        return _check_crag_decide(case)
    if check == "feedback_detect":
        return await _check_feedback_detect(case)
    raise ValueError(f"Unknown agent check {check!r}")


def _check_graph_nodes(case: dict[str, Any]) -> dict[str, Any]:
    class _FakeLLM:
        pass

    graph = build_agent(
        AgentConfig(llm=_FakeLLM()),  # type: ignore[arg-type]
        pattern=case["pattern"],
    )
    nodes = set(graph.get_graph().nodes.keys())
    missing = [n for n in case.get("must_include", []) if n not in nodes]
    forbidden = [n for n in case.get("must_exclude", []) if n in nodes]
    ok = not missing and not forbidden
    return {
        "case_id": case["case_id"],
        "ok": ok,
        "missing": missing,
        "forbidden_present": forbidden,
    }


async def _check_self_rag_pre(case: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = {
        "messages": [HumanMessage(content=case["message"])],
        "metadata": {},
    }
    result = await self_rag_pre_node(
        state,
        llm=None,
        classify_fn=None,
        max_rag_attempts=2,
    )
    meta = result["metadata"]
    expected = case.get("expected_metadata", {})
    mismatches = {
        key: (meta.get(key), expected[key])
        for key in expected
        if meta.get(key) != expected[key]
    }
    return {
        "case_id": case["case_id"],
        "ok": not mismatches,
        "mismatches": mismatches,
    }


def _check_crag_decide(case: dict[str, Any]) -> dict[str, Any]:
    labels = [{"index": i, "label": label} for i, label in enumerate(case["labels"])]
    verdict = compute_verdict(labels)
    action = decide_action(
        verdict,
        attempt=1,
        max_attempts=2,
        web_enabled=False,
        web_used=False,
    )
    expected = case["expected_action"]
    return {
        "case_id": case["case_id"],
        "ok": action == expected,
        "verdict": verdict,
        "action": action,
        "expected_action": expected,
    }


async def _check_feedback_detect(case: dict[str, Any]) -> dict[str, Any]:
    mock_llm = MagicMock()
    mock_llm.arequest_llm = AsyncMock(
        return_value=_mock_llm_response(case["mock_detect"])
    )
    state: AgentState = {
        "messages": [HumanMessage(content=case["message"])],
        "metadata": {},
    }
    result = await detect_feedback_node(state, llm=mock_llm, detect_fn=None)
    route = route_after_detect(result)
    expected_route = case["expected_route"]
    detected = result["metadata"].get("feedback_detected")
    ok = route == expected_route and detected is case["mock_detect"].get("detected")
    return {
        "case_id": case["case_id"],
        "ok": ok,
        "route": route,
        "expected_route": expected_route,
        "feedback_detected": detected,
    }
