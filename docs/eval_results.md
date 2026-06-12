# Eval Results

> 自动/手动评测记录。Profile 定义与本周计划见 [WEEKLY_EVAL_PLAN.md](./WEEKLY_EVAL_PLAN.md)。

## 本地 Gold（Codex 中文文档）

| Date | Profile | Recall@3 | Cases | Notes |
|------|---------|----------|-------|-------|
| 2026-06-12 | baseline..full | 1.00 | 1 | smoke，Docker Qdrant（127.0.0.1:6333） |
| 2026-06-12 | baseline..full | 1.00 | 1 | smoke，`--in-memory` 本地 Qdrant（eval/.cache/） |

## BEIR scifact（500 corpus / 100 queries）

| Date | Profile | Recall@5 | nDCG@10 | Notes |
|------|---------|----------|---------|-------|
| — | — | — | — | 待 Day 3 首跑 |

## Agent E2E

| Date | Pattern | Mock pass | API pass (n/N) | Notes |
|------|---------|-----------|----------------|-------|
| 2026-06-12 | react + 3 agentic | 4/4 | —/— | `python -m eval.run_agent --dataset smoke` |

## 失败 Case 速记

（实验过程中按 query / case id 记录）

## 周结论

（周日填写）

## Smoke (smoke) — 2026-06-12 — Docker

> `python -m eval.run_smoke --dataset smoke --recreate`（无 `--in-memory`）

| Profile | Recall@3 | Chunks | Index OK |
|---------|----------|--------|----------|
| baseline | 1.00 | 41 | yes |
| contextual | 1.00 | 41 | yes |
| s2b | 1.00 | 41 | yes |
| predict_q | 1.00 | 41 | yes |
| full | 1.00 | 41 | yes |

## Smoke (smoke) — 2026-06-12 — 本地缓存

> `python -m eval.run_smoke --dataset smoke --in-memory --recreate`

| Profile | Recall@3 | Chunks | Index OK |
|---------|----------|--------|----------|
| baseline | 1.00 | 41 | yes |
| contextual | 1.00 | 41 | yes |
| s2b | 1.00 | 41 | yes |
| predict_q | 1.00 | 41 | yes |
| full | 1.00 | 41 | yes |
