# LangGraph Server / Studio 部署指南（骨架）

> **状态：** 设计稿 / 待落地。本文描述如何把本仓库的 `build_agent(pattern=...)` 托管到 LangGraph Server，并用 LangSmith Studio 作为聊天客户端。  
> **前提：** 接受绑定 LangGraph 运行时；RAG 仍经 `ToolBox` + `RAG_search_tool`，反思链（CRAG / Self-RAG / Feedback）随整张图一起部署。  
> **相关：** [`FRAMEWORK_DESIGN.md`](FRAMEWORK_DESIGN.md) · [`REFLECTION_GRAPH_DESIGN.md`](REFLECTION_GRAPH_DESIGN.md)

---

## 1. 目标与边界

| 目标 | 说明 |
|------|------|
| 托管完整 Agent 图 | `agent/graph.py` → `build_agent(AgentConfig, pattern=...)` |
| 多轮对话 | Checkpointer + `thread_id`（由 Server / Studio 管理） |
| 可插拔反思模式 | `react` / `react_crag` / `react_self_rag` / `react_all` 等 |
| RAG 检索 | 启动时 `bind_rag_context(...)`，运行时 LLM 调 `RAG_search_tool` |

| 不在本文范围（另见集成文档） | 说明 |
|------------------------------|------|
| 只暴露 RAG、不接 LangGraph | → Open WebUI / MCP / HTTP 薄封装 |
| 替换本仓库 Agent 图 | → 外部 Agent + `RAG_search_tool` 即可 |

---

## 2. 架构一览

```text
┌─────────────────────────────────────────────────────────────┐
│  LangSmith Studio（客户端）                                   │
│  thread_id · 多轮 messages · 可视化图 / 中断 / 调试            │
└───────────────────────────┬─────────────────────────────────┘
                            │ LangGraph API
┌───────────────────────────▼─────────────────────────────────┐
│  LangGraph Server（本仓库 graph factory）                     │
│  build_agent → llm ⇄ tools ⇄ crag / self_rag / feedback     │
└───────────────────────────┬─────────────────────────────────┘
                            │ ToolBox.ainvoke
┌───────────────────────────▼─────────────────────────────────┐
│  tools/LocalTool/RAG_tool.py                                │
│  bind_rag_context → rag/（Qdrant · embed · retrieve）        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 待新增文件（落地时）

> 以下为计划路径，**当前仓库尚未包含**；实现时按此骨架补齐。

| 文件 | 职责 |
|------|------|
| `langgraph.json` | Server 配置：graph 入口、依赖、可选 store / checkpointer TTL |
| `agent/server.py`（或 `get_start/langgraph_app.py`） | **Graph factory**：`make_graph(config)`，供 Server 加载 |
| `.env` | 沿用 [`.env.example`](../.env.example) 中的 `LLM_*` / `EMBEDDING_*` 等 |

### 3.1 `langgraph.json`（草案）

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "agentic_rag": "./agent/server.py:make_graph"
  },
  "env": ".env"
}
```

<!-- TODO: 若需跨 thread 长期记忆，在此增加 store.index 配置 -->

### 3.2 Graph factory 职责（草案）

`make_graph` 在 **进程启动 / 首次执行** 时至少完成：

1. 加载 `.env`（`python-dotenv`）
2. `bind_rag_context(collection=..., in_memory=..., use_small_to_big=..., ...)`
3. 构造 `ToolBox()`（按需缩小 `packages`，省略未用的 MCP）
4. 构造 `LLMClient()`
5. 读取部署配置：`pattern`、`enable_*`、`max_rag_attempts`
6. `return build_agent(AgentConfig(..., checkpointer=???), pattern=...)`

<!-- TODO: checkpointer 选型 — 本地 dev 用 MemorySaver；生产用 Postgres / SQLite -->

<!-- TODO: pattern 从环境变量读取，例如 AGENT_PATTERN=react_crag -->

---

## 4. 环境变量

### 4.1 已有（见 `.env.example`）

| 变量 | 用途 |
|------|------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID` | Agent 与各反思节点 |
| `EMBEDDING_*` | RAG 建库与检索 |
| `BOCHA_API_KEY` 等 | 可选 MCP（CRAG web 兜底） |

### 4.2 部署专用（待定义）

| 变量 | 建议默认 | 说明 |
|------|----------|------|
| `AGENT_PATTERN` | `react` | 对应 `build_agent(pattern=...)` |
| `RAG_COLLECTION` | `rag` | `bind_rag_context` |
| `RAG_IN_MEMORY` | `false` | 生产接 Qdrant；本地可 `true` |
| `RAG_USE_SMALL_TO_BIG` | `false` | 索引耦合项，绑定时固定 |
| `RAG_USE_CONTEXTUAL` | `false` | 同上 |
| `MAX_RAG_ATTEMPTS` | （见 `AgentMetadata` 默认） | Self-RAG / CRAG 重试上限 |
| `LANGSMITH_API_KEY` | — | Studio 追踪（可选但推荐） |

---

## 5. 本地开发流程（待验证）

### 5.1 安装 CLI

```bash
pip install -U langgraph-cli
# 或与 langgraph 版本对齐的 langgraph-cli[inmem]
```

### 5.2 启动 Server

```bash
# 仓库根目录
langgraph dev
```

预期：本地 API（默认 `http://127.0.0.1:2024`）加载 `agentic_rag` graph。

### 5.3 打开 Studio

- 按 LangSmith 文档连接本地 Server，或
- 使用 CLI 输出的 Studio 链接

在 Studio 中：

- 选择 assistant / graph：`agentic_rag`
- 新建 thread → 多轮对话自动带 `thread_id`
- 观察节点：`llm` → `tools` → `crag_eval` / `self_rag_*` / `detect_feedback` 等

---

## 6. Pattern 与运行时行为

| `AGENT_PATTERN` | 节点概览 | 适用场景 |
|-----------------|----------|----------|
| `react` | `llm ⇄ tools` | 基线 ReAct + RAG 工具 |
| `react_crag` | + `crag_eval` 子图 | 检索质量纠正 |
| `react_self_rag` | + `self_rag_pre/post` | 按需检索 + groundedness |
| `react_feedback` | + `detect_feedback` / `plan_feedback` | 用户纠错入口 |
| `react_full` | CRAG + Self-RAG | |
| `react_all` | 三者全开 | 完整 Agentic-RAG |

<!-- TODO: 各 pattern 在 Studio 里的典型 trace 截图 / 示例 thread -->

---

## 7. 多轮会话与上下文

| 机制 | 本仓库字段 | Server 侧 |
|------|-----------|-----------|
| 对话历史 | `AgentState.messages` + `add_messages` | Checkpointer 按 `thread_id` 持久化 |
| 反思元数据 | `AgentState.metadata` | 同 thread 内累积；不直接进 LLM prompt |
| RAG 重试计数 | `metadata.rag_attempt` | 单次 invoke 或同 thread 内（实现时二选一并文档化） |

<!-- TODO: 明确 rag_attempt 是 per-invoke 还是 per-thread，与 REFLECTION_GRAPH_DESIGN 对齐 -->

### 7.1 上下文窗口（后续）

- 在 `llm_node` 调用前 `trim_messages`（按 token 截断）
- 或 Summarization middleware

<!-- TODO: 超长对话策略单独开一节 -->

---

## 8. RAG 索引与数据

| 阶段 | 做法 |
|------|------|
| **建库（离线）** | `get_start/rag_demo.py`、脚本调 `RAG_index_tool` 或 `RAGIndexer.aindex` |
| **Server 运行时** | 只读检索；`bind_rag_context` 指向已有 collection |
| **内存模式** | 仅适合 dev；进程重启数据丢失 |

<!-- TODO: docker-compose Qdrant 与 RAG_COLLECTION 命名约定 -->

---

## 9. 生产部署（提纲）

| 项 | 选项 |
|----|------|
| 托管 | LangGraph Cloud / 自建 Agent Server |
| Checkpointer | Postgres（`langgraph-checkpoint-postgres`） |
| 向量库 | Qdrant（持久化 volume） |
| 密钥 | 环境变量 / Secret Manager，勿写入 `langgraph.json` |
| 观测 | LangSmith tracing |

<!-- TODO: 健康检查、graph rebuild（MCP 工具懒加载）、资源限制 -->

---

## 10. 验收清单

- [ ] `langgraph dev` 能启动，无 import 错误
- [ ] Studio 单轮：`react` + `RAG_search_tool` 返回检索并生成回答
- [ ] Studio 多轮：同 `thread_id` 第二轮能引用上一轮内容
- [ ] `react_crag`：工具返回后经 `crag_eval` 裁剪再回 `llm`
- [ ] `react_self_rag`：`pre` 跳过检索 / `post` 触发重试（受 `max_rag_attempts` 限制）
- [ ] `react_feedback`：纠错类输入走 `plan_feedback`
- [ ] 重启 Server 后 thread 历史仍可恢复（checkpointer 生效）
- [ ] LangSmith 上可见完整 trace

---

## 11. 与「只拼 RAG 到外部 Agent」的对比

| 维度 | LangGraph Server + Studio（本文） | 外部 Agent + `RAG_search_tool` |
|------|-----------------------------------|--------------------------------|
| 客户端 | Studio / LangGraph SDK | Open WebUI、Dify 等 |
| CRAG / Self-RAG / Feedback | ✅ 原样 | ❌ 需自研或放弃 |
| 绑定程度 | LangGraph 运行时 | 仅工具契约 |
| 适合阶段 | Agentic-RAG 联调与演示 | RAG 质量 eval、快速 UI |

---

## 12. 落地任务拆分（Backlog）

| 优先级 | 任务 |
|--------|------|
| P0 | 新增 `agent/server.py`：`make_graph` + `bind_rag_context` bootstrap |
| P0 | 新增 `langgraph.json` |
| P0 | 本地 `langgraph dev` + Studio 冒烟 |
| P1 | Checkpointer（Postgres）与 Qdrant 联调 |
| P1 | `AGENT_PATTERN` 等环境变量约定写进 `.env.example` |
| P2 | `trim_messages` / 多轮 query 改写 |
| P2 | MCP 工具在 Server 环境下的连接与 teardown（参考 LangGraph `make_graph` context manager 模式） |

---

## 13. 参考链接

- [LangGraph persistence / checkpointer](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Add memory (trim / delete messages)](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph Server / CLI](https://docs.langchain.com/langsmith/cli)
- [LangSmith Studio](https://docs.langchain.com/langsmith/studio)
