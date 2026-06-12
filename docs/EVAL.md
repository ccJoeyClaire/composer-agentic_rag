# Eval 文档

> **原则：** Eval 与 Test 分离。pytest 只跑 `tests/`；评测跑 `python -m eval.*`。

## 三层对比

| 层级 | 工具 | 回答的问题 |
|------|------|------------|
| Unit / Integration | `pytest -c tests/pytest.ini` | 有没有写坏？ |
| **Eval** | `python -m eval.run_smoke` 等 | 配置好不好？ |
| 人工 | `get_start/rag_demo.py` | 肉眼对比 |

Eval **不进 CI**（需 API key、有费用、可能 flaky）。

## RAG Profile（smoke）

| profile_id | index contextual | index s2b | index predict_q | search hyde | search reranker |
|------------|------------------|-----------|-----------------|-------------|-----------------|
| baseline | | | | | |
| contextual | yes | | | | |
| s2b | | yes | | | |
| predict_q | | | yes | | |
| full | yes | yes | | yes | yes |

Collection 命名：`eval_{dataset}_{profile_id}`。

## 命令速查

```powershell
# 日常改代码（无 API）
pytest -c tests/pytest.ini -m "not slow and not requires_api"

# Smoke RAG（需 embedding API + Qdrant）
python -m eval.run_smoke --dataset smoke --recreate

# 仅 Agent mock
python -m eval.run_agent --dataset smoke
```

## Gold 格式

**gold_rag.jsonl：**

```json
{
  "query": "问题",
  "expected_keywords": ["词1", "词2"],
  "expected_source": "codex.md",
  "expected_heading_contains": ""
}
```

**gold_agent.jsonl：** 见 `eval/datasets/smoke/gold_agent.jsonl`，字段 `check` 决定断言类型。

## 指标

M1 使用启发式 **Recall@k**：top-k chunk 中是否命中至少一半 `expected_keywords`（或 heading 子串）。

实现：[eval/metrics/recall.py](../eval/metrics/recall.py)

## 配套

- [eval/README.md](../eval/README.md) — 目录与命令
- [eval_results.md](./eval_results.md) — 跑分记录
- [TESTING.md](./TESTING.md) — pytest  only
