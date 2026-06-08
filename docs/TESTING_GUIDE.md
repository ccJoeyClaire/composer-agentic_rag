# RagPipeLine 测试编写指南

> **读者：** 为本仓库贡献或扩展测试的开发者。  
> **配套文档：** [TESTING.md](./TESTING.md)（结构、命令、覆盖清单）。

---

## 1. 三层测试：不要混为一谈

| 层级 | 目的 | 工具 | 断言什么 |
|------|------|------|----------|
| **Unit** | 模块契约、边界 case | `pytest` + `@pytest.mark.unit` | 函数输出、metadata 字段、排序 |
| **Integration** | 多组件串联 | `pytest` + `@pytest.mark.integration` | index → query 能跑通、命中合理 |
| **Eval** | 检索/Agent **好不好** | `gold_*.jsonl` + 指标脚本 | Recall@k、MRR、是否调用 tool |

**规则：** pytest 证明「没写坏」；eval 证明「变好了」。两者都需要，但不能用 unit 测试替代 eval。

---

## 2. 写 Unit 测试的原则

### 2.1 优先测纯函数

无 I/O、无 API 的逻辑最适合 unit：

- `rag/document_augmentation/parent_builder.py`：`merge_windows`、`cluster_overlapping_hits`、`materialize_parent_content`
- `rag/document_augmentation/context_enricher.py`：`build_contextual_header`、`build_embed_text`
- `rag/chunker/semantic_chunker.py`：`SemanticChunker.run`

模板：

```python
import pytest

pytestmark = pytest.mark.unit


def test_something():
    ...
```

### 2.2 Async 测试

异步函数使用 `@pytest.mark.asyncio`（已在 `tests/pytest.ini` 配置 `asyncio_mode = auto`）：

```python
@pytest.mark.asyncio
async def test_async_enricher():
    enricher = ContextualEnricher()
    out = await enricher.aenrich_for_index([...], source="doc.md")
    assert out[0].metadata["embed_text"]
```

### 2.3 构造 Chunk 数据

使用 `tests/conftest.py` 中的工厂，避免在各文件重复：

```python
from tests.conftest import make_chunk, make_small_chunks

chunks = make_small_chunks(["a", "b", "c"], source="kb.md")
```

索引侧 Small-to-Big 测试前记得调用 `assign_parent_chunks(...)`，与生产 `RAGIndexer` 一致。

---

## 3. Integration 测试指南

### 3.1 何时写 integration

- 验证 `RAGIndexer.aindex` → `QdrantVectorStore` → `RAGRetriever.aquery` 链路
- 验证 embedder 与 store 维度一致
- **不要**在 integration 里断言「答案语义正确」（那是 eval）

### 3.2 MockEmbedder + test fake vector store

`conftest.py` 提供：

- `mock_embedder`：相同字符串 → 相同向量，便于「query 与文档句一致则命中」
- `in_memory_vector_store`：`tests/fakes/vector_store.py` 中的轻量实现（**不依赖 Docker / Qdrant**）

> M1 integration 使用 test-only fake store，避免部分环境下 `QdrantVectorStore(url=":memory:")` 与 qdrant-client 不兼容。真实 Qdrant 联调可作为 M2 可选 `@pytest.mark.integration` 用例。

示例见 `tests/rag/test_pipeline_integration.py`。

### 3.3 何时 mock store 而非 Qdrant

Small-to-Big 的 `_resolve_members` 只需按 `chunk_id` 回查时，用 `InMemoryChunkStore` 更轻（见 `test_small_to_big.py`）。

---

## 4. Agent 测试指南

### 4.1 测什么

| 测 | 不测 |
|----|------|
| `if_tool_calls` 路由 | LLM 回答质量 |
| `tool_node` 的 ToolMessage 格式 | 真实 OpenAI 调用 |
| `llm_node` 是否传入 `tools` | 多轮对话业务逻辑 |

### 4.2 Mock ToolBox

Agent 节点依赖 `ToolBox.list_tools()` 与 `ToolBox.ainvoke()`。单测中使用 duck-typing fake（不必继承 `ToolBox`）：

```python
class _FakeToolBox:
    def list_tools(self):
        return []

    async def ainvoke(self, name: str, args: dict) -> ToolResult:
        return ToolResult(name=name, args=args, output="ok")
```

参见 `tests/agent/test_graph_and_nodes.py`。

### 4.3 AIMessage 的 tool_calls 格式

与 `agent/nodes.py` 一致：

```python
AIMessage(
    content="",
    tool_calls=[
        {"name": "rag_search", "args": {"query": "x"}, "id": "call_1"},
    ],
)
```

---

## 5. Eval 框架使用（M1 → M2）

### 5.1 黄金集格式（`tests/eval/gold_rag.jsonl`）

每行一个 JSON 对象（JSONL）：

```json
{
  "query": "用户问题",
  "expected_heading_contains": "可选，heading_path 子串",
  "expected_keywords": ["关键词1", "关键词2"],
  "notes": "人工备注"
}
```

可选扩展字段（M2）：

- `expected_source`：如 `"doc.md"`
- `must_not_contain`：负例关键词

### 5.2 Recall@k 启发式（当前实现）

`tests/eval/test_rag_metrics.py` 中：

- `expected_heading_contains` 命中 → 相关
- 或 `expected_keywords` 中至少一半出现在 chunk 正文 / metadata → 相关

这是 **M1 占位启发式**，正式 eval 可换为：

- 人工标注 chunk id
- 或 LLM-as-judge（成本高）

### 5.3 启用主 eval 用例（M2 步骤）

1. 去掉 `test_rag_gold_recall_at_3` 上的 `@pytest.mark.skip`
2. 在测试中索引 demo 文章（可参考 `get_start/rag_demo.py`）
3. 设定阈值，例如 `mean Recall@3 >= 0.6`
4. 本地运行：`pytest -c tests/pytest.ini tests/eval/ -m eval`（需 API key）

### 5.4 输出对比表（建议 M2）

对同一 `gold_rag.jsonl` 跑多 profile：

| Profile | Recall@3 | 备注 |
|---------|----------|------|
| baseline | 0.xx | vector only |
| +contextual | 0.xx | |
| +s2b | 0.xx | |
| +rerank | 0.xx | `@pytest.mark.slow` |

将结果写入 `docs/eval_results.md` 或 README，用于求职展示。

---

## 6. Mock 边界清单

| 组件 | Unit / Integration 建议 | 真实 API 场景 |
|------|-------------------------|---------------|
| `OpenAIEmbedder` | 用 `MockEmbedder` | `@pytest.mark.requires_api` |
| `LLMClient` / HyDE | `AsyncMock` | eval / manual |
| `CrossEncoderReranker` | 暂不测或 `@pytest.mark.slow` | 可选集成 |
| `QdrantVectorStore` | `:memory:` | Docker 6333 可选 |
| `ToolBox` | `_FakeToolBox` 或 mock `ainvoke` | Agent e2e eval |
| MCP wrappers | mock `call_mcp_tool` 或缺 env 路径 | `@pytest.mark.requires_api` 可选 |

**不要**为了测试去改 `rag/` 源码；新 fake 类放在 `tests/conftest.py` 或 `tests/fakes/`。

---

## 7. 新增测试的检查清单

添加 PR 或本地提交前：

- [ ] 为新文件加 `pytestmark = pytest.mark.unit`（或 integration / eval）
- [ ] 测试名表达行为：`test_merge_windows_dedupes` 而非 `test_merge_1`
- [ ] 无 API key 时 `pytest -c tests/pytest.ini -m "unit or integration"` 全绿
- [ ] 若依赖网络 / 模型，加 `slow` 或 `requires_api` 并文档化
- [ ] 更新 [TESTING.md](./TESTING.md) 覆盖表（若新增模块）

---

## 8. 常见问题

### Q: `pytest` 找不到 `rag` 模块？

在仓库**根目录**运行，并指定配置：

```bash
pytest -c tests/pytest.ini
```

`tests/pytest.ini` 中 `pythonpath = ..` 会把仓库根加入 `PYTHONPATH`。

### Q: integration 测试失败，报 Qdrant 相关错误？

确认 `qdrant-client` 已安装；in-memory 模式不需要 Docker。

### Q: 和 `get_start/rag_demo.py` 的关系？

- demo：人工演示、肉眼对比
- tests：自动化回归
- eval：量化指标；M2 可复用 demo 的 `_build_semantic_pipeline` 建库

### Q: 为什么不测 `CrossEncoderReranker`？

M1 范围刻意排除慢测试与模型下载。添加时使用：

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_rerank_orders_by_relevance():
    ...
```

---

## 9. 推荐阅读顺序

1. 跑通 [TESTING.md](./TESTING.md) 中的日常命令  
2. 阅读 `tests/rag/test_parent_builder.py`（典型 unit）  
3. 阅读 `tests/rag/test_pipeline_integration.py`（典型 integration）  
4. 扩展 `tests/eval/gold_rag.jsonl` 并启用 M2 eval  
