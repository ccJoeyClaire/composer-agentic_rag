# Long-term Planning

> **用途：** 跨 sprint 的技术债与落地 backlog 总表。细项设计见各专题文档；本文只做优先级、依赖与验收口径。  
> **维护：** 完成项打 `[x]`，新增债项注明来源（对话 / 某 doc / issue）。

**相关文档：** [`LANGGRAPH_DEPLOY.md`](LANGGRAPH_DEPLOY.md) · [`FRAMEWORK_DESIGN.md`](FRAMEWORK_DESIGN.md) · [`REFLECTION_GRAPH_DESIGN.md`](REFLECTION_GRAPH_DESIGN.md) · [`rag_pattern.md`](rag_pattern.md)

---

## 1. 总览

| 轨道 | 主题 | 当前状态 |
|------|------|----------|
| **A** | RAG 部署架构（`RagToolContext`、Federated RAG） | 能跑；分层与多租户未做 |
| **B** | LangGraph Server / Studio 托管 | 设计稿；代码未落地 |
| **C** | Agent / 反思链产品化 | 图已接；若干契约与格式待硬化 |
| **D** | RAG 管线进阶（`rag_pattern` Phase 1+） | P0 大部分已有；BM25、GraphRAG 等 roadmap |

---

## 2. 轨道 A — RAG 部署架构与技术债

### 2.1 背景（为何列入长期项）

当前 `tools/LocalTool/RAG_tool.py` 同时承担三类职责：

1. **部署绑定** — `RagToolContext`、`bind_rag_context`、模块级 `_context` 单例  
2. **运行时解析** — `resolve_retriever` / `resolve_indexer`、变体缓存  
3. **LLM Tool 表面** — `@local_tool` 的 `RAG_index_tool` / `RAG_search_tool`

这与 `tools/` 层「薄适配、无全局基础设施状态」的目标不一致；Graph / 终端用户也**无法**在运行时定义 `RagToolContext`（仅启动脚本可 `bind_rag_context`）。

索引期 enrich（contextual、predict-questions、small-to-big metadata）与查询期开关（HyDE、rerank）在概念上已分清，但代码与文档仍偶发混用「索引耦合必须严格对齐」等过时表述，需随重构一并修正。

### 2.2 Federated RAG（多库 / 多租户 / 可注入部署上下文）

> 此处 **Federated RAG** 指：不同 collection、不同 enrich 配置、不同 allow-range 可在**同一进程**内按部署单元或请求路由，而不是全局单例 `_context`。

| ID | 任务 | 优先级 | 说明 |
|----|------|--------|------|
| A-1 | 将 `RagToolContext` + `bind_rag_context` 迁至 `rag/context.py`（或 `agent/rag_bootstrap.py`） | P1 | Tool 文件只保留 `@local_tool` 与调用逻辑 |
| A-2 | 去掉模块级 `global _context`；改为构造时注入（`ToolBox` / `make_graph` / 工厂） | P1 | 依赖 A-1；单测不再 patch `RAG_tool._context` |
| A-3 | `AgentConfig` 或 `make_graph(config)` 显式接收 RAG 部署配置 | P1 | Graph factory 成为唯一 bootstrap 入口 |
| A-4 | 支持多 `collection` 路由（per-tenant / per-assistant） | P2 | 需 A-2；可能引入 `RagToolContext` 注册表或 request-scoped context |
| A-5 | 文档：索引 enrich vs 查询开关的分层（修正「必须预测检索方式」类表述） | P2 | 对齐 `rag/core.py` 与 small-to-big 降级行为 |
| A-6 | `use_predict_questions` 建库策略：默认离线、运行时 `allow_predict_questions` 门禁 | P2 | 避免同 collection 混用不同 enrich |
| A-7 | `RAG_tool.py` 保留 `bind_rag_context` 的 **re-export**（过渡期） | P1 | 避免破坏 `README` / 现有脚本 import 路径 |
| A-8 | 更新 `FRAMEWORK_DESIGN.md` §5.3 与架构图（Context 不在 `tools/`） | P2 | 依赖 A-1 落地后 |

**验收（轨道 A 最小闭环）：**

- [ ] `from rag.context import bind_rag_context` 为推荐路径；旧 import 仍可用  
- [ ] `tests/tools/test_rag_tool.py` 通过且不依赖模块单例 hack  
- [ ] `make_graph` 内完成 bind，Graph 代码不 import `tools.LocalTool.RAG_tool` 做部署  

### 2.3 与 Tool 契约相关的债

| ID | 任务 | 优先级 | 说明 |
|----|------|--------|------|
| A-9 | `RAG_search_tool` 支持 `format="json"` 返回 `[{content, score, metadata}]` | P2 | 见 `REFLECTION_GRAPH_DESIGN.md` §11；降低 CRAG `---` 解析脆性 |
| A-10 | Agent 运行时默认不暴露 `RAG_index_tool`（仅离线建库脚本注册） | P3 | 与「检索在建库之后」的生产模型一致 |
| A-11 | 统一 `rag_attempt` 语义：per-invoke vs per-thread | P1 | 与 `LANGGRAPH_DEPLOY.md` §7、`REFLECTION_GRAPH_DESIGN` 对齐后写死 |

---

## 3. 轨道 B — LangGraph Server / Studio（来自 `LANGGRAPH_DEPLOY.md`）

> 状态：**设计稿 / 待落地**。完整步骤与 env 表见 [`LANGGRAPH_DEPLOY.md`](LANGGRAPH_DEPLOY.md)。

### 3.1 P0 — 本地可演示

| ID | 任务 | 状态 |
|----|------|------|
| B-1 | 新增 `agent/server.py`：`make_graph` + `bind_rag_context` bootstrap | [ ] |
| B-2 | 新增 `langgraph.json`（graph 入口 `agentic_rag`） | [ ] |
| B-3 | 本地 `langgraph dev` + Studio 冒烟（单轮 `react` + `RAG_search_tool`） | [ ] |

**`make_graph` 最低职责（草案）：**

1. `python-dotenv` 加载 `.env`  
2. `bind_rag_context(collection=..., in_memory=..., ...)`（轨道 A 落地后改从 `rag.context`）  
3. `ToolBox()`（可按需缩小 `packages`，省略未用 MCP）  
4. `LLMClient()` + `AgentConfig` + `build_agent(..., pattern=...)`  
5. 返回 compiled graph（checkpointer 见 B-4）

### 3.2 P1 — 可联调环境

| ID | 任务 | 状态 |
|----|------|------|
| B-4 | Checkpointer：dev `MemorySaver`；联调 Postgres / SQLite | [ ] |
| B-5 | `AGENT_PATTERN`、`RAG_COLLECTION`、`RAG_IN_MEMORY`、`RAG_USE_*`、`MAX_RAG_ATTEMPTS` 写入 `.env.example` | [ ] |
| B-6 | docker-compose：Qdrant + `RAG_COLLECTION` 命名约定 | [ ] |
| B-7 | Studio 多 pattern 冒烟：`react_crag` / `react_self_rag` / `react_feedback` / `react_all` | [ ] |

### 3.3 P2 — 生产向

| ID | 任务 | 状态 |
|----|------|------|
| B-8 | `llm_node` 前 `trim_messages` 或 summarization middleware | [ ] |
| B-9 | MCP 在 Server 下的连接生命周期（`make_graph` context manager / teardown） | [ ] |
| B-10 | 健康检查、graph rebuild、资源限制 | [ ] |
| B-11 | LangSmith trace 与各 pattern 示例 thread / 截图归档 | [ ] |
| B-12 | 跨 thread 长期记忆（`langgraph.json` store.index，若需要） | [ ] |

### 3.4 验收清单（摘自 `LANGGRAPH_DEPLOY.md` §10）

- [ ] `langgraph dev` 能启动，无 import 错误  
- [ ] Studio 单轮：`react` + `RAG_search_tool` 返回检索并生成回答  
- [ ] Studio 多轮：同 `thread_id` 第二轮能引用上一轮内容  
- [ ] `react_crag`：工具返回后经 `crag_eval` 裁剪再回 `llm`  
- [ ] `react_self_rag`：`pre` 跳过检索 / `post` 触发重试（受 `max_rag_attempts` 限制）  
- [ ] `react_feedback`：纠错类输入走 `plan_feedback`  
- [ ] 重启 Server 后 thread 历史仍可恢复（checkpointer 生效）  
- [ ] LangSmith 上可见完整 trace  

### 3.5 轨道 A × B 依赖

```text
B-1 make_graph  ──depends──▶  A-1/A-3（推荐：bootstrap 时 bind，不留在 tools 单例）
B-3 冒烟        ──depends──▶  B-1, B-2 + 已有 Qdrant 或 RAG_IN_MEMORY=true
```

---

## 4. 轨道 C — Agent / 反思链（节选）

| ID | 任务 | 优先级 | 来源 |
|----|------|--------|------|
| C-1 | `build_ReAct_agent` → `build_agent(pattern=...)` 命名与 `agent/__init__.py` 导出统一 | P2 | `FRAMEWORK_DESIGN.md` Phase 3 |
| C-2 | `LLMClient` 流式输出 | P3 | Phase 3 |
| C-3 | `agent/harness/` — lifecycle、callbacks | P3 | Phase 3 |
| C-4 | CRAG web 兜底（`bocha` MCP）与 `CragConfig.web_enabled` 生产配置 | P2 | `REFLECTION_GRAPH_DESIGN` |
| C-5 | 反思子图单测矩阵补全（mock `tool_box`，无真实 Qdrant） | P2 | `TEST_CATALOG.md` |

---

## 5. 轨道 D — RAG 管线进阶（节选）

| ID | 任务 | 优先级 | 来源 |
|----|------|--------|------|
| D-1 | `HybridRetriever` BM25 融合（`hybrid_retriever.py` TODO） | P2 | 代码 TODO |
| D-2 | `query_transformer/hybe.py` → `hyde.py` 重命名 | P3 | `FRAMEWORK_DESIGN` |
| D-3 | Optional `rag_node`（Pattern B：检索作为图节点） | P3 | Roadmap |
| D-4 | GraphRAG / 专用 loader（`rag_pattern` P2+） | P3 | `rag_pattern.md` |

---

## 6. 建议执行顺序

```text
阶段 1（可演示）
  B-1 → B-2 → B-3
  并行：A-1 → A-2 → A-7（最小 Context 拆分，供 make_graph 使用）

阶段 2（契约硬化）
  A-11、B-5、B-4、B-7
  A-9（若 CRAG 解析成为瓶颈）

阶段 3（生产 / 多租户）
  B-6、B-8～B-10
  A-3 → A-4（Federated RAG）

阶段 4（产品化与进阶 RAG）
  C-*、D-* 按 eval 与业务优先级拣选
```

---

## 7. 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-11 | 初版：轨道 A（RagToolContext / Federated RAG）、轨道 B（`LANGGRAPH_DEPLOY` backlog）、C/D 节选 |
