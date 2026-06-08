# RAG — 从哪里开始

## 最小可用路径（推荐）

只需要：**分块 → 向量 → Qdrant → 检索**

```python
import asyncio
from rag.simple import build_simple_rag

async def main():
    rag = build_simple_rag(in_memory=True)  # 无需 Docker
    await rag.indexer.aindex("# 标题\n\n正文...", source="doc.md")
    chunks = await rag.retriever.aquery("你的问题", top_k=3)
    for c in chunks:
        print(c.score, c.content[:200])

asyncio.run(main())
```

或运行：

```bash
python get_start/rag_demo.py
```

环境变量（`.env`）：

| 变量 | 说明 |
|------|------|
| `EMBEDDING_API_KEY` 或 `LLM_API_KEY` | Embedding API |
| `EMBEDDING_BASE_URL` 或 `LLM_BASE_URL` | 兼容 OpenAI 的 base URL |
| `EMBEDDING_MODEL_ID` | 默认 `text-embedding-3-small` |
| `EMBEDDING_BATCH_SIZE` | 单次请求条数上限（未设置时：DashScope 自动 `10`，其它默认 `2048`） |

## 进阶（可选，不必一开始就用）

| 目录 | 用途 |
|------|------|
| `core.py` | `RAGIndexer`（建库）、`RAGRetriever`（查询）、`RAGPipeline`（门面） |
| `base.py` | 含 `BaseContextualEnricher` 等 ABC |

HyDE / hybrid / reranker / contextual 通过 `RAGRetriever` / `RAGIndexer` 构造参数注入。反思、GraphRAG、领域 prompt 在 **Agent** 层扩展。

**原则：** 只查已有库时用 `build_retriever()`（无需 chunker）；建库 + 查询用 `build_simple_rag()` 或 `build_simple_pipeline()`。
