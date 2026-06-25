# README 骨架（composer-agentic_rag）

> 本文件是 **README 写作框架**，不是最终对外文档。写完后把内容迁到根目录 `README.md`。
>
> **当前项目状态（对齐用）**
>
> | 模块 | 状态 | README 里怎么写 |
> |------|------|-----------------|
> | `rag/` | ✅ 基本完工 | 作为稳定能力完整介绍 |
> | `get_start/` | ✅ RAG 示例可跑；Agent 示例 🚧 今日计划补 | RAG 路径写实；Agent 标 WIP 或占位 |
> | `agent/` | 🚧 重构中 | 说明是**推荐方向**，API 可能变动 |
> | `agent/` | legacy | 一句带过或「迁移中」 |
> | 评测 | 📋 计划：Easy Dataset 出题 + RAGChecker | 留空节 + 复现命令占位；**不写** BEIR / `_eval_` 为主路径 |
>
> 行首标记：`[填]` 必写 · `[选]` 可选 · `[暂缓]` 等有结果再写 · `[WIP]` 重构未完成

---

## 0. 怎么用这份骨架

1. 从上到下复制到 `README.md`。
2. 删掉每节下方的「▸ 本章放什么」说明块（或保留在 PR 里对照）。
3. `[填]` 先写 RAG + get_start；Agent / 评测节可保留占位。
4. 子文档（RAG 开关详解、Agent 设计）链到 `docs/`，README 只放摘要。

---

# composer-agentic_rag

`[填]` **一句话**：组件化 Agentic RAG——可插拔检索管线 + LangGraph ReAct Agent，检索增强与 Agent 能力均按需开关组合。

`[填]` **状态徽章（二选一或组合）**：

- RAG：**可用**
- Agent（`agent`）：**重构中**
- 系统化评测：**计划中**（Easy Dataset + RAGChecker）

`[选]` 可选：CI badge、Python 版本、License badge。

---

▸ **本章放什么**

- 读者 10 秒内知道：这是什么、哪部分能直接用、哪部分还在动。
- **不要**在这里展开 CRAG 节点、chunker 实现细节。
- 若对外开源，第二句可写「和通用 RAG 库的差异」（Agent 解耦、profile 开关、能力插件化）。

---

## 特性概览

`[填]` 3～5 条 bullet，按**读者价值**写，不是按目录写：

```markdown
- **RAG 可独立使用**：建库 / 检索与 Agent 解耦，`rag/` 单独 import 即可
- **Profile 开关组合**：chunk 策略、contextual、s2b、HyDE、rerank 等通过 `arg_config.yaml` profile 切换
- **Agent 能力插件**（`agent`，重构中）：Retrieval Gate、RAG Profile Router、Human Feedback 等独立 toggle
- **可观测**：index / retrieve 可 dump JSONL trace（见 `get_start/`）
- **评测路线**（计划中）：Easy Dataset 生成 QA + RAGChecker 评答案质量
```

`[暂缓]` 量化数字（Recall、RAGChecker 分数）——有跑分后再加一条。

---

▸ **本章放什么**

- 卖点清单；每条最好能对应「用户能做什么」。
- Agent 相关 bullet 加「重构中」以免 API 变动被当成 bug。

---

## 架构

`[填]` 保留分层 ASCII 图，**路径要对齐现状**：

```
用户 query ──▶ agent/  (LangGraph ReAct + 可选 capabilities)   [WIP]
                    │ ToolBox
                    ▼
              tools/  (RAG_search · web · MCP …)
                    │
                    ▼
              rag/    (index 管线 · retrieve 管线 · Qdrant)
                    │
              llm/    (OpenAI 兼容 client，贯穿各层)
```

`[填]` 图下 2～3 句：**依赖方向**（Agent 不直接 import retriever，经 ToolBox）、**配置入口**（RAG → `arg_config.yaml`；Agent → `AgentConfig`）。

`[选]` Mermaid 版（与 ASCII 二选一，避免重复）。

---

▸ **本章放什么**

- 一张图 + 短说明即可。
- 若 `agent/` 仍保留，脚注：`agent/` 为旧版，新开发请用 `agent/`。

---

## 快速开始

`[填]` 前置条件清单：

| 条件 | RAG demo | Agent demo（日后） |
|------|:--------:|:------------------:|
| Python 3.x + `pip install -r requirements.txt` | ✓ | ✓ |
| `.env`（Embedding；HyDE/rerank 等还需 LLM key） | ✓ | ✓ |
| Qdrant `127.0.0.1:6333` | ✓（当前 get_start） | 视 Agent 示例而定 |
| Docker | 选 | 选 |

### 安装

`[填]`

```bash
pip install -r requirements.txt
copy .env.example .env   # Windows；Unix: cp .env.example .env
```

### 环境变量

`[填]` **精简表**（完整列表链 `.env.example`）：

| 变量 | 用途 |
|------|------|
| `EMBEDDING_*` | 建库与向量检索 |
| `LLM_*` | Agent、HyDE、contextual enrich 等 |
| `BOCHA_API_KEY` / `TAVILY_*` | 可选联网搜索 |
| … | … |

### RAG：索引 → 检索（推荐第一步）

`[填]` 对齐 `get_start/` 真实命令：

```bash
# 1. 索引 fixture 文章 → Qdrant，输出 runs/index.jsonl
python -m get_start.index_example

# 2. 同一 collection 检索，输出 runs/retrieve.jsonl
python -m get_start.retrieve_example
```

`[填]` **预期结果**（1～2 句）：例如 retrieve 终端打印 top-k chunk 摘要；`get_start/runs/*.jsonl` 含 trace。

`[填]` **可调旋钮**（各 1 句 + 链接）：改 `_PROFILE_ID` / `arg_config.yaml` profile；改 `_QUERY`。

### Agent：最小示例

`[WIP]` 今日补 `get_start/agent_example.py` 后再填。

> **状态：重构中。** 以下 API 以 `agent` 为准，可能变动。

```python
# TODO: agent build_agent 最小 invoke 示例
# 前置：bind_rag_context(...) 或共享 collection
```

`[暂缓]` 直到 get_start Agent 示例稳定再写完整代码块。

### 不用 Docker / 内存模式（可选路径）

`[选]` 若仍支持 `in_memory=True`，给 5～10 行 `build_RAG_indexer` / `build_RAG_retriever` 片段；注明与 get_start（Qdrant）二选一。

---

▸ **本章放什么**

- **可复制粘贴能跑**的路径；RAG 写满，Agent 允许占位。
- 每个命令注明：需要什么 key、要不要先跑上一步。
- 旧 README 里的 `rag_demo.py` **已不存在**——以 `index_example` / `retrieve_example` 为准。

---

## RAG 配置与 Profile

`[填]` **概念一句**：profile 是索引期 + 检索期开关的命名组合，定义在 `arg_config.yaml`。

`[填]` **简表**（不必列全部分 profile，选 3～5 个代表）：

| profile_id | 典型用途 | 索引侧 | 检索侧 |
|------------|----------|--------|--------|
| `baseline` | 默认上手 | token chunk | 向量 top-k |
| `baseline_hyde` | 查询扩展 | … | HyDE |
| `full` | 全开 ablation | contextual + s2b + … | hyde + rerank |
| … | … | … | … |

`[填]` 切换方式：

```python
profile = get_profile(get_rag_config(), "baseline")
```

`[填]` **深链**（子文档占位，文件可后续补）：

- Profile 字段说明 → `docs/rag_pattern.md`（或新建 `docs/RAG.md`）
- 组件清单（chunker / store / reranker）→ 同上

---

▸ **本章放什么**

- 帮助读者「改哪个 yaml、换哪个 profile」，不是实现原理。
- 完整 ablation 表放 eval 文档，不堆在 README。

---

## Agent（agent）

`[WIP]` 本节在 Agent 重构稳定前保持「方向说明 + 占位示例」。

### 设计要点

`[填]` 2～4 句：

- LLM 仍是决策中心；capabilities 只追加节点/工具，**不是**互斥的 `pattern=` preset。
- 与 `tools/` 的 `ToolBox` 集成；RAG 工具名、web 工具名可配置。

### Capability 开关

`[填]` 表（对齐 `agent/config.py`）：

| 配置项 | 作用 | 默认 |
|--------|------|------|
| `enable_retrieval_gate` | 检索结果质量门控，再回 LLM | off |
| `enable_rag_profile_router` | 按 query 选 RAG profile | off |
| `enable_human_feedback` | 人机反馈 + checkpoint | off |
| `enable_web_search` | 暴露联网工具 | on |

### 构建示例

`[WIP]` get_start Agent 示例就绪后替换：

```python
# from agent.builder import build_agent
# from agent.config import AgentConfig
# ...
```

### 与旧版 `agent/` 的关系

`[填]` 一句：旧图（CRAG / Self-RAG / Feedback pattern）在 `legacy/agent/`；逻辑已迁移到 `agent/` capabilities 中，**新代码勿依赖 `legacy/agent/graph.py`**。

`[填]` 设计深链 → `docs/FRAMEWORK_DESIGN.md`、`docs/REFLECTION_GRAPH_DESIGN.md`（注明部分描述针对 legacy，阅前看日期）。

---

▸ **本章放什么**

- 稳定后：10～20 行可运行示例 + capability 表。
- 重构中：表可以写，代码块用 TODO；避免文档与 `builder.py` 漂移。

---

## 工具系统

`[填]` `ToolBox` 自动发现 `@local_tool` / `@mcp_tool`。

| 工具 | 类型 | 作用 |
|------|------|------|
| `RAG_search_tool` / `RAG_index_tool` | local | 检索 / 入库 |
| web search | mcp | Bocha / Tavily |
| markitdown | mcp | 文档 → Markdown |

`[选]` 新增本地工具：在 `tools/LocalTool/` 加函数 + 装饰器，一行说明即可。

---

▸ **本章放什么**

- Agent 节已讲 RAG 绑定；这里只列**工具清单**和扩展方式。

---

## 评测

`[暂缓]` **有 Easy Dataset + RAGChecker 结果后再填摘要表**；现阶段写「路线 + 占位命令」。

### 评测路线（计划）

`[填]` 文字说明（**不要**以 BEIR / `_eval_/rag_eval` 为主叙事）：

1. **出题**：Easy Dataset 基于语料生成 QA（export JSONL）。
2. **跑候选答案**：Agent 或 RAG 管线产出 candidates。
3. **评分**：RAGChecker（+ 可选 rubric checklist）。

`[填]` 与 pytest 边界一句：单元测试在 `tests/`；离线评测需 API key，**不进 CI**。

### 复现命令（占位）

`[暂缓]` 命令随脚本落地再填，例如：

```bash
# TODO: Easy Dataset export → enrich gold → run candidates → RAGChecker
# python -m ... 
```

### 结果摘要

`[暂缓]`

```markdown
<!-- 示例结构，有数据后再打开
| 配置 | RAGChecker 均分 | 备注 |
|------|-----------------|------|
| baseline | — | |
| full | — | |
-->
```

`[填]` 完整记录链到 **`docs/eval_results.md`**（新建或沿用，与 README 摘要分离）。

### 历史 / 内部 eval

`[选]` 若保留 `_eval_`（BEIR、HotpotQA 等）仅供开发参考，**单独一小节**注明「内部/历史，非推荐路径」，避免与 Easy Dataset 主线混淆。

---

▸ **本章放什么**

- 现在：讲清**评什么、怎么评、结果放哪**；表格留空即可。
- 将来：README 只放 1 个小表 + 复现三行命令；图表、ablation 进 `docs/eval_results.md`。

---

## 项目结构

`[填]` 简表（一行职责，不展开文件树）：

| 目录 | 职责 | 状态 |
|------|------|------|
| `rag/` | 建库 + 检索 + Qdrant | ✅ |
| `get_start/` | 上手示例与 `runs/` trace | ✅ RAG / 🚧 Agent |
| `agent/` | LangGraph Agent + capabilities | 🚧 |
| `agent/` | 旧版 ReAct + 反思 pattern | legacy |
| `tools/` | ToolBox、local/MCP 工具 | ✅ |
| `llm/` | OpenAI 兼容 client | ✅ |
| `tests/` | pytest | ✅ |
| `docs/` | 设计、测试、评测文档 | 持续 |
| `_eval_/` | 历史 BEIR 等离线 eval | 内部参考，非主评测路径 |

---

▸ **本章放什么**

- 新人找代码用；**状态列**减少误用 legacy / WIP 模块。

---

## 测试

`[填]`

```bash
pytest -c tests/pytest.ini -m "not slow and not requires_api"   # 日常
pytest                                                           # 全量
```

`[填]` 链接：`docs/TESTING.md`、`docs/TESTING_GUIDE.md`。

---

▸ **本章放什么**

- 命令 + 文档链接；marker 细节不进 README。

---

## 文档索引

`[填]` 表格，只列**读者真的会点的**：

| 文档 | 内容 |
|------|------|
| `docs/README_SKELETON.md` | 本 README 写作框架 |
| `docs/rag_pattern.md` | RAG 模式与 profile 说明 |
| `docs/FRAMEWORK_DESIGN.md` | 整体框架设计 |
| `docs/REFLECTION_GRAPH_DESIGN.md` | 反思 / gate 设计（部分 legacy） |
| `docs/TESTING.md` | pytest 说明 |
| `docs/eval_results.md` | 评测跑分记录（待 Easy Dataset 结果） |
| `docs/LANGGRAPH_DEPLOY.md` | 部署（若适用） |

---

▸ **本章放什么**

- 索引表；避免 README 正文重复子文档长文。

---

## License / Acknowledgments

`[选]` License 文件名。

`[选]` 致谢：CRAG、Self-RAG、Easy Dataset、RAGChecker、BEIR（若用过）等。

---

▸ **本章放什么**

- 开源必备元信息；个人项目可极简。

---

## 附录：从骨架到发布 README 的检查清单

- [ ] 顶部状态与真实能力一致（RAG ✅ / Agent 🚧 / Eval 📋）
- [ ] 快速开始命令在本机跑通并贴预期输出
- [ ] 无失效路径（`rag_demo.py`、`rag/README.md`、`eval/` 等）
- [ ] Agent 示例与 `agent` API 一致
- [ ] 评测节不以 `_eval_` 为主；Easy Dataset + RAGChecker 占位已留
- [ ] 深链文档存在或标「待写」
- [ ] 删掉本骨架里的「▸ 本章放什么」说明块
