from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.query_transformer.hyde import append_hyde_log_entry


def test_append_hyde_log_entry_writes_one_json_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "hyde.jsonl"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HYDE_LOG", "1")
    monkeypatch.setenv("HYDE_LOG_PATH", str(log_path))
    monkeypatch.setenv("HYDE_LOG_PROFILE", "baseline_hyde")
    monkeypatch.setenv("HYDE_LOG_COLLECTION", "pooleval_nfcorpus_baseline_hyde")

    append_hyde_log_entry(query="q1", hyde_document="hypothetical passage")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["query"] == "q1"
    assert entry["hyde_document"] == "hypothetical passage"
    assert entry["profile"] == "baseline_hyde"
    assert entry["collection"] == "pooleval_nfcorpus_baseline_hyde"
    assert entry["timestamp_utc"]


@pytest.mark.asyncio
async def test_hyde_transform_appends_log_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag.query_transformer.hyde import HyDETransformer

    log_path = tmp_path / "hyde.jsonl"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HYDE_LOG", "1")
    monkeypatch.setenv("HYDE_LOG_PATH", str(log_path))

    class _FakeLLM:
        async def arequest_llm(self, messages, *, temperature: float = 0.0):
            class _Resp:
                content = "generated hyde doc"

            return _Resp()

    transformer = HyDETransformer(llm_client=_FakeLLM())  # type: ignore[arg-type]
    result = await transformer.atransform("what is RAG?")

    assert result == "generated hyde doc"
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1
