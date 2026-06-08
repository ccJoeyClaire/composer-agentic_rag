# RAG Pattern 路线图

> **用途：** 为后续开发做铺垫，与仓库内 `rag/` 模块对齐。  
> **语言：** 中文为主，术语附英文。  
> **场景：** 多格式文档（PDF / 网页 / Markdown）→ 检索增强问答；反思类能力偏 **Agent 图** 实现。  
> **基线分块：** 默认 **semantic chunking**（`semantic_chunker.py`）；`simple.py` 仍用 `MarkdownChunker`。

---

## 1. 文档约定（每条 pattern 的字段）

| 字段 | 说明 |
|------|------|
| **ID** | 本文编号 |
| **阶段** | `index` / `query` / `generate` / `agent` |
| **当前实现** | 仓库里已有的路径或类 |
| **目标模块** | 计划放置位置（可与当前不同） |
| **Pipeline 参数** | 构造 `RagPipeline` 的 kwargs；无则写 `—` |
| **依赖** | 必须先具备的 pattern ID |
| **优先级** | P0（先做）/ P1（Phase 1）/ P2 / P3 |
| **参考** | 论文、官方文档（待补链接） |

**编排原则（与代码一致）：**

- **`RagPipeline`（`rag/core.py`）：** 固定流水线内的 `aindex` / `aquery`（transform、retrieve、rerank、compress…）。
- **Agent（LangGraph）：** Self-RAG、Feedback、CRAG 等「反思与纠错」族 + 何时调用 `RagPipeline.as_tool()`。
- **入口：** 日常开发先 `rag/simple.py`；进阶再挂可选构造参数（如 `contextual_enricher`）。

---

## 2. 与现有架构的对照

```
rag/simple.py          → 最小闭环（P0 基座）
rag/core.py            → RagPipeline 中枢（aindex / aquery / as_tool）
rag/base.py            → 含 BaseContextualEnricher；GraphRAG/反思在 retriever 或 agent
```

| 本文档概念 | 代码映射 |
|------------|----------|
| 固定查询流水线 | `RagPipeline._run_query` in `rag/core.py` |
| Agent 决定是否检索 | LangGraph ReAct + `RagPipeline.as_tool()` |
| 向量 + BM25 融合 | `HybridRetriever`（构造参数，RRF 待实现） |
| HyDE | `query_transformer/`（构造参数） |
| 上下文增强族（4/5/6） | `contextual_enricher=` + `BaseContextualEnricher` |
| 知识图谱 | `rag/retriever/` 或 graph store（roadmap，非 profile hook） |
| 反思族（11/12/17） | **Agent 层**（`agent/`，非 `RagPipeline`） |

---

## 3. Phase 规划

| 阶段 | 范围 | 说明 |
|------|------|------|
| **基座** | #1 | 跑通 index → query（`build_simple_pipeline`） |
| **P0** | #3, #7, #8, #10, #15 | 文档 + 实现优先（约 5 条） |
| **Phase 1（P1）** | **#2 semantic chunking** 优先；其余 P1 待定（如 #16 fusion、#4–6 族） | 在 P0 之后 1–2 个月；**已确认：Phase 1 第一条做 semantic chunking** |
| **P2+** | 其余 | 文档先行，接口已预留 |

---

## 4. Pattern 清单

### 索引侧（Index）

#### 1. Simple RAG

| 项 | 内容 |
|----|------|
| **阶段** | index + query |
| **说明** | 分块 → embedding → 向量库 → 向量检索；无额外增强。 |
| **当前实现** | `rag/simple.py`, `MarkdownChunker`, `OpenAIEmbedder`, `QdrantVectorStore`, `VectorRetriever` |
| **目标模块** | 同上（保持为默认基座） |
| **RagProfile** | `—`（基座无 tag） |
| **依赖** | — |
| **优先级** | 基座（先于 P0） |
| **参考** | — |

---

#### 2. Semantic Chunking

| 项 | 内容 |
|----|------|
| **阶段** | index |
| **说明** | 按语义边界分块（非固定 token 窗口）；**目标为默认分块策略**。 |
| **当前实现** | `rag/chunker/md_chunker.py`（token + 标题，非语义） |
| **目标模块** | `rag/chunker/semantic_chunker.py` |
| **RagProfile** | `—`（chunker 层能力，可不进 profile） |
| **依赖** | #1 |
| **优先级** | **P1（Phase 1 首选）** — 在 P0 完成后第一个做的增强 |
| **参考** | 待补 |

---

#### 3. Small-to-Big

| 项 | 内容 |
|----|------|
| **阶段** | index + query |
| **说明** | **索引小块**（用于精确向量匹配），**返回 / 生成时拼成大块**（更多上下文）。与 #14 不同：不要求预先有 summary 层。 |
| **当前实现** | `rag/retriever/small_to_big_retriever.py`（空） |
| **目标模块** | `rag/chunker/` 存 small + parent_id；`rag/retriever/small_to_big_retriever.py` |
| **RagProfile** | `—` |
| **依赖** | #1, #2 推荐 |
| **优先级** | **P0** |
| **参考** | 待补 |

---

#### 4–6. 上下文增强族（Context Enrichment Family）

> 你确认为 **同一族**：索引阶段增强 chunk 的可检索上下文。对应代码方向：`BaseContextualEnricher` / `CONTEXTUAL_RETRIEVAL`。

| ID | 名称 | 侧重点 |
|----|------|--------|
| 4 | Context-enriched retrieval | 检索时用 enriched 表示（如 embedding 含上下文） |
| 5 | Contextual chunk headers | 块前加标题/面包屑式 header（Anthropic 语境） |
| 6 | Document augmentation | 文档级增强（摘要、关键词、QA 对等写入 payload） |

| 项 | 内容 |
|----|------|
| **阶段** | index（主）, query（可选） |
| **当前实现** | `rag/base.py` → `BaseContextualEnricher`；`rag/document_augmentation/context_enricher.py` |
| **目标模块** | `rag/document_augmentation/context_enricher.py` 或 `rag/chunker/contextual_headers.py` |
| **Pipeline 参数** | `contextual_enricher=ContextualEnricher()` |
| **依赖** | #1 |
| **优先级** | P1（族内可先实现 #5 headers，再 #4/#6） |
| **参考** | Anthropic Contextual Retrieval；待补链接 |

---

#### 14. Hierarchical Indices

| 项 | 内容 |
|----|------|
| **阶段** | index + query |
| **说明** | **先检索大块 summary**，锁定范围后再 **精细检索 chunk**。依赖预先生成 summary 层。与 #3 区别：层次索引 + 两阶段检索，而非 small 命中后扩成大段。 |
| **当前实现** | — |
| **目标模块** | `rag/document_augmentation/summary_index.py`, `rag/retriever/hierarchical_retriever.py` |
| **RagProfile** | `—`（`small_to_big_parent_tokens` 构造参数） |
| **依赖** | #1, summary 生成（LLM） |
| **优先级** | P2 |
| **参考** | RAPTOR / 层级索引；待补 |

---

### 查询侧（Query）

#### 7. Query Transformation

| 项 | 内容 |
|----|------|
| **阶段** | query |
| **说明** | 改写、扩展、多查询（Multi-Query）、Step-back 等。 |
| **当前实现** | `rag/query_transformer/hybe.py`（HyDE stub）, `query_rewriter.py`（空） |
| **目标模块** | `rag/query_transformer/` |
| **RagProfile** | `—`（`query_transformer` 构造参数） |
| **依赖** | #1 |
| **优先级** | **P0** |
| **参考** | 待补 |

---

#### 15. HyDE

| 项 | 内容 |
|----|------|
| **阶段** | query |
| **说明** | LLM 生成 **假设性答案**，用其向量检索真实文档。 |
| **当前实现** | `HyDETransformer`（stub） |
| **目标模块** | `rag/query_transformer/hyde.py`（rename from `hybe.py`） |
| **RagProfile** | `—`（`HyDETransformer` 构造参数） |
| **依赖** | #7, `LLMClient` |
| **优先级** | **P0** |
| **参考** | HyDE paper；待补 |

---

#### 16. Fusion（RRF）

| 项 | 内容 |
|----|------|
| **阶段** | query |
| **说明** | **向量检索 + BM25**，经 **RRF** 融合；与 #8 不同，不做 LLM 判别。 |
| **当前实现** | `HybridRetriever`（暂委托 `VectorRetriever`，无 BM25/RRF） |
| **目标模块** | `rag/retriever/hybrid_retriever.py` + `rank-bm25` |
| **RagProfile** | `—`（`HybridRetriever` 构造参数） |
| **依赖** | #1 |
| **优先级** | P1（与 P0 的 #8 串行配合） |
| **参考** | RRF；待补 |

---

#### 8. Reranker（Cross-Encoder 精排）

| 项 | 内容 |
|----|------|
| **阶段** | query |
| **说明** | 向量（或 #16 RRF）**粗召回 Top-N**（如 50）→ **Cross-Encoder** 对每条候选打相关分 → 按分数取 **Top-K**。与 #16 串行：**先融合召回，再精排**。 |
| **默认算法（目标）** | 见下方 §4.1 |
| **当前实现** | `rag/reranker/cross_encoder_reranker.py`（stub） |
| **目标模块** | **`rag/reranker/cross_encoder_reranker.py`**（P0，默认）；`llm_pairwise_reranker.py`（P2 可选） |
| **RagProfile** | `—`（pipeline 可选 `reranker: BaseReranker`） |
| **依赖** | #1；`sentence-transformers` 或同类；推荐在 #16 之后 |
| **优先级** | **P0** |
| **参考** | `cross-encoder/ms-marco-MiniLM-L-6-v2`；待补 |

##### §4.1 Reranker 算法说明（默认：Cross-Encoder）

**默认：Cross-Encoder 对 Top-N 逐条打分 → 排序 → Top-K**

```text
输入: query Q, 候选 chunks [c1..cN]（N≈50）, 目标 K
模型: Cross-Encoder（如 ms-marco-MiniLM-L-6-v2）

流程:
  1. 对每个 ci，构造输入 pair (Q, ci.content)
  2. 模型输出相关分数 score_i
  3. 按 score 降序排序，取 Top-K 写入 Chunk.score

复杂度: N 次小模型前向（可 batch）；可本地 GPU/CPU，无 LLM API 成本
推荐库: sentence-transformers CrossEncoder
```

| 方案 | 原理 | 成本 | 本项目 |
|------|------|------|--------|
| **Cross-Encoder（默认）** | `[query; chunk]` 联合编码，相关分更准 | 低、可本地、可 batch | **P0 实现** |
| **LLM pairwise（备选）** | LLM 两两比较 / 锦标赛当裁判 | 高（多次 API） | **P2 可选**，见 §4.2 |
| **Bi-Encoder（向量）** | query/chunk 各 embed 再算相似度 | 最低 | 已在 #1 检索阶段使用，不重复为 rerank |

**超参（待实现时填默认值）：** `recall_n=50`, `top_k=5`, `cross_encoder_model=…`, `batch_size=…`

##### §4.2 LLM pairwise 精排（备选）

当 Cross-Encoder 分数接近、或领域极偏、需要强推理时，可切换 **LLM 作裁判**：

```text
LLM 读 Q 与两个 chunk → 判哪个更相关 → 锦标赛 / 淘汰 → Top-K
成本高，仅作 P2 增强或「低置信度回退」
```

与 Cross-Encoder 的区别：LLM 是 **生成式裁判**；Cross-Encoder 是 **判别式打分器**，更适合固定流水线默认路径。

##### §4.3 与 Bi-Encoder 的关系（三层检索）

| 层级 | 方法 | 作用 |
|------|------|------|
| 召回 | Bi-Encoder（`embedder` + Qdrant） | 从全库快速捞 Top-N |
| 精排 | **Cross-Encoder（默认）** | 在 N 条里重排序 |
| 可选 | LLM pairwise | 疑难 query 再精炼 |

---

#### 9. RSE（Relevant Segment Extraction）

| 项 | 内容 |
|----|------|
| **阶段** | query |
| **说明** | 相似度过滤 → 上下文窗口扩展 → 片段打分 → 阈值筛选。 |
| **当前实现** | — |
| **目标模块** | `rag/post_retrieval/rse.py` |
| **RagProfile** | `—` |
| **依赖** | #1, #8 推荐 |
| **优先级** | P2 |
| **参考** | 待补 |

---

#### 10. Contextual Compression

| 项 | 内容 |
|----|------|
| **阶段** | query（检索后、进 LLM 前） |
| **说明** | 压缩召回上下文，降低 token、去噪。 |
| **当前实现** | — |
| **目标模块** | `rag/post_retrieval/contextual_compression.py` |
| **RagProfile** | `—` |
| **依赖** | #1 |
| **优先级** | **P0** |
| **参考** | LangChain ContextualCompressionRetriever；待补 |

---

### 生成与 Agent 侧

#### 11–12–17. 反思与纠错族（Reflection Family）

> 合并为一族，**仅在 Agent 层**实现（已从 `RagPipeline` 移除）。  
> **落地设计：** 见 [`agent/REFLECTION_GRAPH_DESIGN.md`](agent/REFLECTION_GRAPH_DESIGN.md)（子图 + metadata + RAG-as-tool）。

| ID | 名称 | 侧重点 |
|----|------|--------|
| 11 | Feedback Loop | 用户/系统反馈驱动再检索或再生成 |
| 12 | Self-RAG | 生成后自检：是否需要 retrieve / 答案是否 grounded |
| 17 | CRAG | 召回后 **相关性评估**（correct / incorrect / ambiguous），决定重搜或降级 |

| 项 | 内容 |
|----|------|
| **阶段** | agent（主）, query（评估节点） |
| **当前实现** | `agent/graph.py`（ReAct）；反思节点 roadmap |
| **目标模块** | `agent/nodes/rag_reflect.py` |
| **RagProfile** | `—`（Agent state / metadata） |
| **依赖** | #1, LangGraph ReAct + `as_tool()` |
| **优先级** | P2（文档 P0 未选，但需预留） |
| **参考** | Self-RAG, CRAG papers；待补 |

---

#### 13. Knowledge Graph

| 项 | 内容 |
|----|------|
| **阶段** | index + query |
| **说明** | 实体关系图谱增强检索与推理。 |
| **当前实现** | — |
| **目标模块** | `rag/retriever/` graph retriever, `rag/store/` |
| **RagProfile** | `—` |
| **依赖** | #1 |
| **优先级** | P3 |
| **参考** | GraphRAG；待补 |

---

## 5. 推荐流水线（P0 完成后）

```text
[Index]
  semantic chunking (#2) → optional context family (#4–6) → embed → Qdrant
  optional: small-to-big parent links (#3)

[Query — RagPipeline._run_query]
  query transformation (#7) → HyDE (#15) optional
  → fusion RRF (#16) OR vector only
  → cross-encoder rerank (#8)  // 默认；可选 LLM pairwise（P2）
  → contextual compression (#10)

[Phase 1 — Index]
  semantic chunking (#2) 替换/并列 token 分块
  → (optional) RSE (#9)

[Generate / Agent]
  Agent 调用 as_tool 或注入上下文 → Self-RAG / CRAG / Feedback (#11–12–17)
```

---

## 6. 多格式文档（场景备注）

| 格式 | 建议前置模块 | 关联 pattern |
|------|----------------|--------------|
| Markdown | 现有 `MarkdownChunker` | #1, #2 |
| PDF | `rag/loaders/pdf_loader.py`（待建） | #1 |
| 网页 | `rag/loaders/web_loader.py`（roadmap） | #1 |

---

## 7. 待你补充（下一轮可填）

- [ ] 各 **参考** 列的具体 URL
- [x] Phase 1 首选：**#2 semantic chunking**
- [ ] Phase 1 其余条目（如 #16 fusion、#4–6 族）排序
- [ ] CRAG 在 Agent 层用 state metadata 或专用 node 表达
- [x] Reranker 默认：**Cross-Encoder**；LLM pairwise 为 P2 备选
- [ ] Cross-Encoder：`recall_n`、`top_k`、`model_name`、`batch_size`

---

## 8. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-27 | 初版结构化：问卷结果、与 `rag/` 映射、P0/Phase1 范围 |
| 2026-05-27 | Phase 1 确认 #2 semantic；#8 曾记 LLM pairwise，后改回 **Cross-Encoder 为默认** |
