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
  eval/
    gold_rag.jsonl        # RAG 黄金集（M1 示例 3 条）
    test_rag_metrics.py   # Recall@k 框架 + helper 单测
docs/
  TESTING.md              # 本文档
  TESTING_GUIDE.md        # 编写测试的指导
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
pytest -c tests/pytest.ini tests/rag tests/agent tests/tools tests/eval -m "unit or integration"

# 全部（不含 eval / slow / requires_api 时可加 -m "not eval"）
pytest -c tests/pytest.ini

# 仅 RAG 模块
pytest -c tests/pytest.ini tests/rag/

# 仅 Agent 模块
pytest -c tests/pytest.ini tests/agent/

# eval 框架（M1 主 eval 用例默认 skip，helper 单测会跑）
pytest -c tests/pytest.ini tests/eval/ -m "unit or eval"

# 快速 CI（排除慢测试与 API）
pytest -c tests/pytest.ini -m "not slow and not requires_api and not eval"
```

---

## 4. Marker 说明

定义见 `tests/pytest.ini`：

| Marker | 含义 | 典型用例 |
|--------|------|----------|
| `unit` | 纯逻辑，无外部服务 | `parent_builder`、`semantic_chunker` |
| `integration` | 流水线 wiring（test fake store 或 Qdrant） | `test_pipeline_integration.py` |
| `eval` | 检索 / Agent 能力评测 | `test_rag_gold_recall_at_3`（M1 skip） |
| `slow` | 下载 Cross-Encoder 等 | 暂未使用，预留给 reranker 测试 |
| `requires_api` | 需要 `EMBEDDING_API_KEY` / `LLM_API_KEY` | eval 端到端 |

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

### Eval（框架）

| 文件 | 状态 |
|------|------|
| `eval/gold_rag.jsonl` | 3 条示例 query + `expected_keywords` |
| `eval/test_rag_metrics.py` | `recall_at_k` helper 单测；主 eval 用例 `@pytest.mark.skip` 待 M2 启用 |

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

- [ ] 启用 `test_rag_gold_recall_at_3`，索引 Codex demo 文章并设 Recall@3 阈值
- [ ] baseline vs `+contextual` vs `+s2b` vs `+rerank` 对比表输出
- [ ] BEIR scifact 子集 Recall@5 / nDCG@10
- [ ] Agent eval：`gold_agent.jsonl` + mock/API 端到端断言
- [ ] Cross-Encoder 慢测试（`@pytest.mark.slow`）
