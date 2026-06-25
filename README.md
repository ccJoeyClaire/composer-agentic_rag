# composer-agentic_rag

一个**组件化的 Agentic RAG 框架**：把「可插拔 RAG」与「带反思的 LangGraph ReAct Agent」组合在一起。检索增强（small-to-big / contextual / HyDE / rerank）与智能体反思模式（CRAG / Self-RAG / Feedback）都做成开关，按需启用。

> **状态：开发中。** 核心链路与单元/集成测试已就绪；**系统化 eval 尚未运行**——检索质量与各开关的收益仍待量化（计划见 `docs/WEEKLY_EVAL_PLAN.md`，结果记录在 `docs/eval_results.md`）。当前的默认参数属于「合理默认」，不代表已验证的最优配置。

---

## 分层架构

```
                         ┌─────────────────────────────────────┐
   用户 query ─────────▶ │  agent/  (LangGraph ReAct + 反思)     │
                         │  CRAG · Self-RAG · Feedback           │
                         └───────────────┬─────────────────────┘
                                         │ 通过 ToolBox 调用
                         ┌───────────────▼─────────────────────┐
                         │  tools/  (装饰器自动发现的工具箱)      │
                         │  local: RAG_search / RAG_index / math │
                         │  mcp:   markitdown · web search       │
                         └───────────────┬─────────────────────┘
                                         │ RAG 工具委托
                         ┌───────────────▼─────────────────────┐
                         │  rag/  (index + retrieve 两条流程)     │
                         │  chunk → embed → Qdrant → retrieve    │
                         └─────────────────────────────────────┘
   llm/  : OpenAI 兼容的同步/异步 Chat 客户端，贯穿各层
```

各层只依赖下一层的抽象：Agent 不直接 `import RAGRetriever`，而是经 `ToolBox` 调 `RAG_search_tool`（见 `agent/rag_tool.py`、`tools/LocalTool/RAG_tool.py`）。

---

## 目录结构

| 目录 | 内容 |
|------|------|
| `llm/` | `LLMClient` —— 兼容 OpenAI 的同步/异步 chat 客户端 |
| `tools/` | `ToolBox` 工具运行时 + `@local_tool` / `@mcp_tool` 装饰器自动发现；`LocalTool/`（RAG、math）、`MCPTool/`（markitdown、web search） |
| `rag/` | 可插拔 RAG 模块（建库 + 检索）。详见 [`rag/README.md`](rag/README.md) |
| `agent/` | LangGraph ReAct agent 与反思子图：`graph.py`、`reflection/`（Self-RAG、Feedback）、`subgraph/CRAG.py` |
| `get_start/` | 上手 demo（`rag_demo.py`） |
| `tests/` | 单元 / 集成 / eval 测试；清单见 `docs/TEST_CATALOG.md` |
| `docs/` | 设计与测试/评测文档（见文末索引） |

---

## 快速开始

### 1. 安装与配置

```bash
pip install -r requirements.txt
copy .env.example .env          # 然后填入你的 key（Windows）
```

`.env` 关键变量：

| 变量 | 用途 |
|------|------|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID` | Chat LLM（Agent、HyDE、predict-questions、反思节点） |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL_ID` | Embedding（RAG 建库与检索） |
| `EMBEDDING_BATCH_SIZE` | 单次 embed 请求条数上限（DashScope 建议 `10`） |
| `BOCHA_API_KEY` / `OPENAI_API_KEY`… | 可选：web search / markitdown OCR 等 MCP 工具 |

未配置时 `EMBEDDING_*` 会回退到 `LLM_*`。完整项见 `.env.example`。

### 2. 跑 RAG demo（最快验证）

```bash
python get_start/rag_demo.py                    # 仅分块检查，不调 API
python get_start/rag_demo.py --rag --in-memory  # 端到端检索（需 Embedding key，无需 Docker）
```

### 3. 用 RAG 库（无需 Docker，内存模式）

```python
import asyncio
from rag import build_RAG_indexer, build_RAG_retriever

async def main():
    indexer = build_RAG_indexer("demo", in_memory=True)
    retriever = build_RAG_retriever(
        "demo", in_memory=True,
        store=indexer.store, embedder=indexer.embedder,   # 内存模式须共享 store
    )
    await indexer.aindex("# 标题\n\n正文……", source="doc.md")
    for c in await retriever.aquery("你的问题", top_k=3):
        print(round(c.score, 4), c.content[:120])

asyncio.run(main())
```

RAG 的全部开关、组件清单与 Qdrant 说明见 [`rag/README.md`](rag/README.md)。

### 4. 构建一个 Agent

```python
import asyncio
from langchain_core.messages import HumanMessage
from legacy.agent.graph import AgentConfig, build_agent
from llm.client import LLMClient
from tools.tool_box import ToolBox

async def main():
    agent = build_agent(
        AgentConfig(llm=LLMClient(), tool_box=ToolBox()),
        pattern="react_all",     # 见下方模式表
    )
    state = await agent.ainvoke({"messages": [HumanMessage(content="你的问题")]})
    print(state["messages"][-1].content)

asyncio.run(main())
```

> 若要让 Agent 使用 RAG 检索工具，需在启动时调用 `tools.LocalTool.RAG_tool.bind_rag_context(collection=..., in_memory=...)` 绑定共享的 store/embedder。

---

## Agent 反思模式

`build_agent(config, pattern=...)` 通过组合三个反思能力得到不同模式：

| pattern | CRAG | Self-RAG | Feedback | 说明 |
|---------|:---:|:---:|:---:|------|
| `react` | | | | 基础 ReAct（LLM ⇄ tools） |
| `react_crag` | ✓ | | | 检索后做 Corrective-RAG 评估再回到 LLM |
| `react_self_rag` | | ✓ | | 检索前判定是否需检索 + 回答后做 groundedness 检查与重试 |
| `react_feedback` | | | ✓ | 入口检测用户反馈并先规划 |
| `react_full` | ✓ | ✓ | | CRAG + Self-RAG |
| `react_all` | ✓ | ✓ | ✓ | 全部启用 |

- **CRAG**（`agent/subgraph/CRAG.py`）：评估检索结果质量，必要时纠正/重查。
- **Self-RAG**（`agent/reflection/self_rag.py`）：`pre` 节点决定是否检索；`post` 节点检查答案是否有据，未达标且仍有 `max_rag_attempts` 配额时回到 LLM 重试。
- **Feedback**（`agent/reflection/feedback.py`）：检测用户反馈并据此规划。

也可不用 pattern，直接在 `AgentConfig` 上设 `enable_crag` / `enable_self_rag` / `enable_feedback`。

---

## 工具系统（`tools/`）

`ToolBox` 在启动时扫描 `tools.LocalTool` 与 `tools.MCPTool` 两个包，自动发现被 `@local_tool` / `@mcp_tool` 装饰的函数，并把它们转成 OpenAI function-calling schema 供 LLM 选用。

| 工具 | 来源 | 作用 |
|------|------|------|
| `RAG_search_tool` / `RAG_index_tool` | local | 检索 / 入库；查询期开关（HyDE、rerank、recall_n、top_k）可在 allow-range 内按调用切换 |
| `math_tool` | local | 数值计算 |
| markitdown | mcp | 文档 → Markdown（可选 OCR） |
| web search | mcp | 联网搜索（Bocha） |

新增本地工具：在 `tools/LocalTool/` 写一个带类型注解的函数并加 `@local_tool` 即可被自动发现。

---

## 测试与评测

```bash
pytest                 # 全量
pytest tests/rag       # 只测 RAG
pytest tests/agent     # 只测 Agent / 反思
```

- 测试结构与 marker：`docs/TESTING.md`；逐条解读：`docs/TEST_CATALOG.md`；如何写测试：`docs/TESTING_GUIDE.md`。
- **评测状态：尚未运行系统化 eval。** 检索质量（命中率 / nDCG / BEIR）与 Agent 端到端表现待量化；计划见 `docs/WEEKLY_EVAL_PLAN.md`，结果记录在 `docs/eval_results.md`。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [`rag/README.md`](rag/README.md) | RAG 模块详解（开关、组件、Qdrant） |
| `docs/FRAMEWORK_DESIGN.md` | 框架整体设计 |
| `docs/REFLECTION_GRAPH_DESIGN.md` | 反思图（CRAG / Self-RAG / Feedback）设计 |
| `docs/rag_pattern.md` | RAG 模式（small-to-big、contextual、HyDE…）说明 |
| `docs/TESTING.md` · `docs/TESTING_GUIDE.md` · `docs/TEST_CATALOG.md` | 测试体系 |
| `docs/WEEKLY_EVAL_PLAN.md` · `docs/eval_results.md` | 评测计划与结果 |
