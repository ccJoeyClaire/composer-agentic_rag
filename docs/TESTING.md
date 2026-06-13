# RagPipeLine 测试文档

> **语言：** 中文为主，术语保留英文（pytest、Recall@k、marker 等）。  
> **原则：** 测试代码只新增在 `tests/`；不修改 `rag/`、`agent/` 等业务模块。

---

## 1. 目录结构

```text
tests/
  pytest.ini              # pytest 配置与 marker 定义
  conftest.py             # 公共 fixtures（MockEmbedder、sample markdown 等）
  rag/
    test_parent_builder.py
    test_small_to_big.py
    test_semantic_chunker.py
    test_contextual_enricher.py
    test_pipeline_integration.py
  agent/
    test_graph_and_nodes.py
  tools/
    test_tool_box.py
    test_mcp_tools.py
eval/                     # 独立评测（不进 pytest），见 docs/EVAL.md
docs/
  TESTING.md              # 本文档
  TESTING_GUIDE.md        # 编写测试的指导
  EVAL.md                 # Eval 命令与 profile
```

---

## 2. 环境准备

```bash
# 仓库根目录
pip install -r requirements.txt
```

依赖包含：`pytest`、`pytest-asyncio`（见 `requirements.txt` 末尾）。

---

## 3. 运行命令

**配置文件位于 `tests/pytest.ini`**，请在仓库根目录执行：

```bash
# 仅 Tools 模块
pytest -c tests/pytest.ini tests/tools/

# 日常：unit + integration（默认推荐）
pytest -c tests/pytest.ini tests/rag tests/agent tests/tools -m "unit or integration"

# 全部
pytest -c tests/pytest.ini

# 仅 RAG 模块
pytest -c tests/pytest.ini tests/rag/

# 仅 Agent 模块
pytest -c tests/pytest.ini tests/agent/

# 快速 CI（排除慢测试与 API）
pytest -c tests/pytest.ini -m "not slow and not requires_api"

# 检索/Agent 能力评测（独立脚本，不进 pytest）
# python -m eval.runners.run_rag
```

---

## 4. Marker 说明

定义见 `tests/pytest.ini`：

| Marker | 含义 | 典型用例 |
|--------|------|----------|
| `unit` | 纯逻辑，无外部服务 | `parent_builder`、`semantic_chunker` |
| `integration` | 流水线 wiring（test fake store 或 Qdrant） | `test_pipeline_integration.py` |
| `slow` | 下载 Cross-Encoder 等 | reranker 相关 |
| `requires_api` | 需要 `EMBEDDING_API_KEY` / `LLM_API_KEY` | 可选集成用例 |

---

## 5. 当前覆盖范围（M1）

### RAG（约 18 个 test function）

| 模块 | 文件 | 覆盖要点 |
|------|------|----------|
| Small-to-Big 索引 | `test_parent_builder.py` | 稳定 `chunk_id`、`anchor_window`、merge、cluster、物化去重 |
| Small-to-Big 检索 | `test_small_to_big.py` | 单 hit / 重叠 / 不重叠、store 回查 member、`recall_multiplier` |
| SemanticChunker | `test_semantic_chunker.py` | `heading_path`、token 上限、语义切分 |
| ContextualEnricher | `test_contextual_enricher.py` | header、`embed_text`、retrieve prepend |
| Pipeline | `test_pipeline_integration.py` | `RAGIndexer` + in-memory Qdrant + `RAGRetriever.aquery` |

### Agent（6 个 test function，其中 3 个可能 skip）

| 模块 | 文件 | 覆盖要点 |
|------|------|----------|
| 路由 | `test_graph_and_nodes.py` | `if_tool_calls` → `tools` / 非 tools（需能 import `agent.graph`） |
| 节点 | `test_graph_and_nodes.py` | `tool_node` 成功 / 错误、`llm_node` 传 tools |

### Tools（10 个 test function）

| 模块 | 文件 | 覆盖要点 |
|------|------|----------|
| ToolBox | `test_tool_box.py` | register、resolve、`ainvoke`、schema、`get_available_tools` |
| MCP wrappers | `test_mcp_tools.py` | registry 路径、缺 API key / MCP 命令时的错误信息 |

### Eval（独立，见 [EVAL.md](./EVAL.md)）

| 位置 | 说明 |
|------|------|
| `eval/rag/` | Index / Search + Recall |
| `eval/agent/` | CRAG context vs baseline |
| `eval/runners/` | smoke 一键编排 |
| `eval/datasets/smoke/` | manifest + gold 数据 |
| `tests/eval/test_agent_gold.py` | Agent 行为 gold（原 run_agent） |

---

## 6. Fixtures（`conftest.py`）

| 名称 | 说明 |
|------|------|
| `mock_embedder` | 确定性 `MockEmbedder`（同文本 → 同向量） |
| `sample_markdown` | 带 `# Alpha` / `# Beta` 的样例 Markdown |
| `in_memory_vector_store` | `tests/fakes/InMemoryVectorStore`（integration 用，无需 Docker） |
| `in_memory_qdrant_store` | 上述 fixture 的别名，保持测试文件兼容 |
| `make_chunk` / `make_small_chunks` | 工厂函数（在测试文件中 `from tests.conftest import ...`） |
| `InMemoryChunkStore` | 轻量 `BaseVectorStore`，用于 S2B member 回查单测 |

---

## 7. 与 CI 的推荐集成

```yaml
# 示例 GitHub Actions step
- run: pip install -r requirements.txt
- run: pytest -c tests/pytest.ini -m "unit or integration" --tb=short
```

M1 **不包含** GitHub Actions 配置文件；可按上式自行添加。

---

## 8. 相关文档

- 如何新增用例、mock 策略、eval 黄金集格式 → [TESTING_GUIDE.md](./TESTING_GUIDE.md)
- RAG pattern 路线图 → 仓库根目录 `rag_pattern.md`

---

## 9. M2 计划（进行中）

> 逐日任务与交付标准见 [WEEKLY_EVAL_PLAN.md](./WEEKLY_EVAL_PLAN.md)；结果记入 [eval_results.md](./eval_results.md)。

- [ ] `python -m eval.runners.run_rag` 全 profile 跑通
- [ ] baseline vs `+contextual` vs `+s2b` vs `+rerank` 对比表写入 eval_results.md
- [ ] BEIR scifact 子集 Recall@5 / nDCG@10
- [ ] Agent eval：`gold_agent.jsonl` + mock/API 端到端断言
- [ ] Cross-Encoder 慢测试（`@pytest.mark.slow`）
