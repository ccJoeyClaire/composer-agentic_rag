# `rag/` — 可插拔的检索增强（RAG）模块

一个组件化的 RAG 工具箱：**建库（index）** 与 **查询（retrieve）** 两条流程，所有增强能力（small-to-big、contextual、HyDE、rerank…）都通过构造参数按需开关。

> 状态：核心链路与单元/集成测试已就绪；**系统化 eval 尚未运行**（见文末「评测状态」）。API 仍可能随评测结论调整。

---

## 两条流程

```
建库 (RAGIndexer.aindex):
  text → chunk → [parent 切分] → [contextual / predict-question 增强] → embed → store → verify

查询 (RAGRetriever.aquery):
  query → [HyDE 改写] → vector recall → [contextual 拼接] → [rerank] → top_k
```

- `RAGIndexer`：离线 / 入库时使用，不含 retriever、reranker、HyDE。
- `RAGRetriever`：只查询，不含 chunker；只需要一个 `BaseRetriever`（通常包裹 embedder + store）。

---

## 快速开始

最小路径，**无需 Docker**（Qdrant 跑在内存里）：

```python
import asyncio
from rag import build_RAG_indexer, build_RAG_retriever

async def main():
    indexer = build_RAG_indexer("demo", in_memory=True)
    retriever = build_RAG_retriever(
        "demo",
        in_memory=True,
        store=indexer.store,        # 复用同一个内存库
        embedder=indexer.embedder,
    )

    await indexer.aindex("# 标题\n\n正文……", source="doc.md")

    chunks = await retriever.aquery("你的问题", top_k=3)
    for c in chunks:
        print(round(c.score, 4), c.content[:200])

asyncio.run(main())
```

> 内存模式下 `indexer` 与 `retriever` 必须共享同一个 `store`（如上传入 `store=indexer.store`），否则查不到数据。连接真实 Qdrant 时则不需要，二者按 `collection` 名对齐即可。

或直接跑现成 demo：

```bash
python get_start/rag_demo.py            # 仅分块检查，不调 API
python get_start/rag_demo.py --rag --in-memory   # 端到端（需 .env 里的 Embedding key）
```

---

## 两个工厂函数

| 函数 | 流程 | 何时用 |
|------|------|--------|
| `build_RAG_indexer(collection, ...)` | chunk → embed → store | 建库 / 文档入库 |
| `build_RAG_retriever(collection, ...)` | transform → retrieve → rerank | 查询已有库 |

只查已有库时，单独 `build_RAG_retriever()` 即可（无需 chunker）。

### 增强开关（默认全关）

| 开关 | indexer | retriever | 说明 |
|------|:---:|:---:|------|
| `use_small_to_big` | ✓ | ✓ | 索引 512-token 小块，检索时返回合并后的 parent 窗口（≈3072 token） |
| `use_contextual` | ✓ | ✓ | 建库时给每个 chunk 加情境化 header 并增强 embed 文本；查询时拼接上下文 |
| `use_predict_questions` | ✓ | — | LLM 为每个 chunk 预生成可能的问题，提升召回 |
| `use_hyde` | — | ✓ | 用 LLM 生成假设性答案再做向量改写（HyDE） |
| `use_reranker` | — | ✓ | CrossEncoder 精排（需 `sentence-transformers`） |
| `recall_n` | — | ✓ | rerank 前的向量召回条数，默认 `50` |

> 启用 `use_small_to_big` / `use_contextual` 的一侧开关，必须在 indexer 与 retriever **两侧一致**，否则 payload 对不上。

---

## 目录结构

| 路径 | 内容 |
|------|------|
| `base.py` | 数据类 `Chunk` / `RagContext` / `RagResult` 与全部 ABC（`BaseChunker`、`BaseEmbedder`、`BaseVectorStore`、`BaseRetriever`、`BaseReranker`、`BaseQueryTransformer`、`BaseContextualEnricher`） |
| `core.py` | `RAGIndexer`、`RAGRetriever` —— 两条流程的编排器 |
| `build.py` | `build_RAG_indexer` / `build_RAG_retriever` —— 按开关组装组件 |
| `chunker/` | `SemanticChunker`（相似度断点）、`MarkdownChunker`（仅 token 预算） |
| `embedder/` | `OpenAIEmbedder`（兼容 OpenAI 的 Embedding API） |
| `store/` | `QdrantVectorStore`（`:memory:` 或 Docker） |
| `retriever/` | `VectorRetriever`、`SmallToBigRetriever`、`HybridRetriever`（向量 + BM25，尚未接入工厂） |
| `reranker/` | `CrossEncoderReranker` |
| `query_transformer/` | `HyDETransformer` |
| `document_augmentation/` | `ContextualEnricher`、`PredictQuestionEnricher`、`parent_builder`（`assign_parent_chunks`） |

---

## 环境变量（`.env`）

| 变量 | 说明 |
|------|------|
| `EMBEDDING_API_KEY` 或 `LLM_API_KEY` | Embedding API key |
| `EMBEDDING_BASE_URL` 或 `LLM_BASE_URL` | 兼容 OpenAI 的 base URL |
| `EMBEDDING_MODEL_ID` | 默认 `text-embedding-3-small` |
| `EMBEDDING_BATCH_SIZE` | 单次请求条数上限（DashScope 自动 `10`，其它默认 `2048`） |

`use_predict_questions` / `use_hyde` 还会调用 Chat LLM，依赖 `LLM_API_KEY` / `LLM_BASE_URL`。

## Qdrant（非内存模式）

```bash
docker compose up -d      # 在 127.0.0.1:6333 起 Qdrant
```

不传 `in_memory=True` 时默认连 `127.0.0.1:6333`。

---

## 与 Agent 层的关系

`rag/` 只负责「分块 → 向量 → 检索」。反思（Self-RAG）、GraphRAG、领域 prompt 等编排在 **`agent/`** 层扩展；Agent 不直接 `import RAGRetriever`，而是通过 `ToolBox` 调用（见 `agent/rag_tool.py`）。`RAGIndexer.as_tool()` / `RAGRetriever.as_tool()` 可把流程包成同步工具函数。

## 测试与评测

- 单元 / 集成测试：`tests/rag/`（运行 `pytest tests/rag`）。测试清单见 `docs/TEST_CATALOG.md`。
- **评测状态：尚未运行系统化 eval。** 检索质量（命中率 / nDCG / BEIR 等）与各增强开关的收益仍待量化，计划见 `docs/WEEKLY_EVAL_PLAN.md`，结果将记录在 `docs/eval_results.md`。在 eval 完成前，上面的默认参数与开关组合属于「合理默认」，不代表已验证的最优配置。
