# Eval（独立评测）

与 `tests/` 中的 pytest **完全分离**。Eval 回答「检索/Agent 好不好」，不进 CI。

## 前置

1. `.env` 配置 `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`（千问 embedding）
2. Docker Qdrant：`docker compose up -d`（默认 `127.0.0.1:6333`）
3. 无 Docker 时加 `--in-memory`（仅当前进程有效）

可选环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `EVAL_INDEX_CONCURRENCY` | 3 | 多文档并行 index |
| `EVAL_LLM_CONCURRENCY` | 15 | predict-questions LLM 并发 |

## Smoke（1 文档 + 1 gold）

```powershell
# 全流程：5 个 RAG profile
python -m eval.run_smoke --dataset smoke

# 分步
python -m eval.run_index --dataset smoke --profile baseline --recreate
python -m eval.run_search --dataset smoke --profile baseline

# Agent mock（无 API）
python -m eval.run_agent --dataset smoke
```

结果追加到 [docs/eval_results.md](../docs/eval_results.md)。

## 目录

```text
eval/datasets/smoke/   manifest + gold_rag + gold_agent
eval/profiles.py       RAG profile 定义
eval/run_index.py      Index 管线
eval/run_search.py     Search 管线
eval/run_smoke.py      编排
eval/run_agent.py      Agent mock eval
```

详见 [docs/EVAL.md](../docs/EVAL.md)。
