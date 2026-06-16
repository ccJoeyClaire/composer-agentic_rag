"""Unit tests for agent context compare (no API)."""

from __future__ import annotations

import pytest

from _eval_.paths import dataset_dir
from tests.eval.agent_context_helpers import (
    load_agent_context_cases,
    load_fixture_text,
    recall_on_context,
    recall_on_texts,
    run_crag_trim,
)


@pytest.fixture
def crag_trim_case() -> dict:
    cases = load_agent_context_cases(
        dataset_dir("smoke") / "gold_agent_context.jsonl"
    )
    return next(c for c in cases if c["case_id"] == "crag_trim_noise_gold")


@pytest.mark.asyncio
async def test_crag_trim_keeps_gold_passage(crag_trim_case: dict) -> None:
    noise = load_fixture_text(crag_trim_case["fixture_noise"])
    gold = load_fixture_text(crag_trim_case["fixture_gold"])
    labels = crag_trim_case["crag_labels"]

    final = await run_crag_trim(
        crag_trim_case,
        noise=noise,
        gold=gold,
        labels=labels,
    )

    assert gold in final
    assert noise not in final


@pytest.mark.asyncio
async def test_crag_trim_improves_recall_over_raw_fixture_at_1(
    crag_trim_case: dict,
) -> None:
    noise = load_fixture_text(crag_trim_case["fixture_noise"])
    gold = load_fixture_text(crag_trim_case["fixture_gold"])
    source = crag_trim_case["expected_source"]

    raw_at_1 = recall_on_texts([noise, gold], crag_trim_case, k=1, source=source)
    final = await run_crag_trim(
        crag_trim_case,
        noise=noise,
        gold=gold,
        labels=crag_trim_case["crag_labels"],
    )
    crag_recall = recall_on_context(final, crag_trim_case, k=1, source=source)

    assert raw_at_1 == 0.0
    assert crag_recall == 1.0
