"""Gold agent behavior cases (pytest; data in _eval_/datasets/smoke/gold_agent.jsonl)."""

from __future__ import annotations

import pytest

from _eval_.paths import dataset_dir
from tests.eval.agent_checks import run_agent_case
from tests.eval.gold_recall import load_gold_cases


def _gold_agent_cases() -> list[dict]:
    return load_gold_cases(dataset_dir("smoke") / "gold_agent.jsonl")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _gold_agent_cases(), ids=lambda c: c["case_id"])
async def test_gold_agent_case(case: dict) -> None:
    result = await run_agent_case(case)
    assert result["ok"], result
