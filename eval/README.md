# eval/ 目录说明

> 与 `tests/` 分离：eval 回答「配置好不好」；tests 回答「有没有写坏」。不进 CI。

## 目录结构

```text
eval/
  bootstrap.py, paths.py, profiles.py
  rag/              run_index.py, run_search.py, metrics/recall.py
  agent/            run_context.py, context.py, compare.py
  runners/          run_rag.py, run_compare.py
  tools/            convert_dataset.py
  loaders/
  datasets/smoke/   manifest, gold, fixtures
```

## 命令

```powershell
python -m eval.runners.run_rag --recreate
python -m eval.rag.run_index --dataset smoke --profile baseline --recreate
python -m eval.rag.run_search --dataset smoke --profile baseline
python -m eval.rag.run_search --dataset smoke --profile baseline --trace
python -m eval.runners.run_compare --recreate
python -m eval.agent.run_context --dataset smoke --profile baseline
pytest tests/eval/test_agent_gold.py -v
python -m eval.tools.convert_dataset --help
```

## 两条 eval 线

| 线 | 模块 | 比什么 |
|----|------|--------|
| **RAG Profile** | `eval/rag/` | 同 query，不同 RAG 参数；无 Agent |
| **反思 Context** | `eval/agent/` | baseline retriever vs CRAG context |

Agent 行为 gold 在 `tests/eval/test_agent_gold.py`（非 eval CLI）。

详见 [docs/EVAL.md](../docs/EVAL.md)。
