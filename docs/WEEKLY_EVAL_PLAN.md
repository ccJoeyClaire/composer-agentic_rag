# Test & Eval 两周计划（修订版）

> **周期：** 2026-06-08 — 2026-06-21（**10 个工作日**，每天 8h，共 **80h**）  
> **修订原因：** `tests/` 为 AI 生成，需先读懂现有 108 个用例再写 eval，原 1 周 56h 不够。  
> **目标：** ① 搞懂现有测试在保护什么 ② RAG eval（本地 gold + BEIR）③ Agent 端到端 eval  
> **配套：** [TEST_CATALOG.md](./TEST_CATALOG.md)（**每个 test 的含义**，必读）· [TESTING.md](./TESTING.md) · [TESTING_GUIDE.md](./TESTING_GUIDE.md)

---

## 0. 时间重估（为什么从 1 周变成 2 周）

| 工作块 | 原估计 | 修订估计 | 说明 |
|--------|--------|----------|------|
| **读懂现有 108 个 test** | 0h（默认已懂） | **14–16h** | 有 [TEST_CATALOG.md](./TEST_CATALOG.md) 可压缩；无目录约 22h |
| RAG 本地 eval + profile | 10h | 10h | 不变 |
| BEIR scifact | 12h | 12h | 不变 |
| Agent mock + API eval | 14h | 14h | 不变 |
| Integration 补洞 | 6h | 6h | 不变 |
| 文档 + 汇总 | 4h | 4h | 不变 |
| **缓冲**（读不懂、返工） | 10h | **8h** | 两日机动 |
| **合计** | 56h | **~74h** | → **10 个工作日**（80h 含缓冲） |

**结论：** 不要跳过「读 test」直接写 eval——否则你不知道新 eval 和现有契约是否重复、冲突。

---

## 0.1 两周要交付什么

| 交付物 | 路径 | 完成标准 |
|--------|------|----------|
| 本地 RAG gold 集 | `tests/eval/gold_rag.jsonl` | ≥ 20 条，覆盖章节/关键词/口语化/负例 |
| RAG eval 用例（可跑） | `tests/eval/test_rag_metrics.py` | 去掉 skip，`test_rag_gold_recall_at_3` 全绿 |
| BEIR scifact eval | `tests/eval/beir_scifact/` + `test_beir_metrics.py` | 子集索引 + 官方 qrels Recall@5 |
| Profile 对比脚本 | `tests/eval/run_profiles.py` | 4 profile 输出 JSON/表格 |
| Agent gold 集 | `tests/eval/gold_agent.jsonl` | ≥ 15 条（tool 行为 + 可选答案关键词） |
| Agent eval 用例 | `tests/eval/test_agent_metrics.py` | mock e2e 全绿；`requires_api` 子集可跑 |
| 结果记录 | `docs/eval_results.md` | 每日更新 baseline / +contextual / +s2b / +full |
| 文档同步 | `docs/TESTING.md` | M2 checklist 勾选、覆盖表更新 |
| **测试解读笔记** | 个人笔记或 `docs/TEST_CATALOG.md` 批注 | 每个模块「保护的核心契约」一行总结 |

---

## 1. 三层时间分配（80h 总预算）

| 层级 | 工时 | 占比 | 回答的问题 |
|------|------|------|------------|
| **Test 导读（读懂现有用例）** | **16h** | 20% | 每个 unittest 在保护什么？改了代码该跑谁？ |
| **Unit / Integration 补洞** | 6h | 8% | HyDE mock、S2B 端到端、reranker smoke |
| **RAG Eval** | 22h | 28% | 本地 gold Recall@k + BEIR scifact Recall@5 |
| **Agent E2E Eval** | 20h | 25% | tool 路由、Self-RAG/CRAG 行为、少量真实 API |
| **汇总 + 文档** | 8h | 10% | eval_results、TESTING 同步 |
| **缓冲** | 8h | 10% | 卡住的 test、API 限速、返工 |

**日常命令（固定习惯）：**

```bash
# 每次改代码前 / 提交前（无 API，< 1 分钟）
pytest -c tests/pytest.ini -m "not slow and not requires_api and not eval" --tb=short

# RAG eval（需 .env embedding key）
pytest -c tests/pytest.ini tests/eval/test_rag_metrics.py -m "eval and requires_api" -v

# Agent eval（mock 子集，无 API）
pytest -c tests/pytest.ini tests/eval/test_agent_metrics.py -m "eval and not requires_api" -v

# 全量 eval（周末汇总）
pytest -c tests/pytest.ini tests/eval/ -m eval -v
```

---

## 2. RAG Profile 定义（全周统一）

本周所有 RAG / BEIR 对比使用同一组 profile，避免混用参数：

| Profile ID | `use_contextual` | `use_small_to_big` | `use_hyde` | `use_reranker` | 用途 |
|------------|------------------|--------------------|------------|----------------|------|
| `baseline` | false | false | false | false | 向量检索下限 |
| `contextual` | true | false | false | false | Contextual header 增益 |
| `s2b` | false | true | false | false | Small-to-big 增益 |
| `full` | true | true | true | true | 生产向全栈（`@pytest.mark.slow`） |

索引参数与 demo 对齐：`SemanticChunker` 512 token / 64 overlap（见 `rag/build.py`）。

---

## 3. BEIR 方案（scifact 子集）

**为何选 scifact：** 体量适中（~5k 段落）、英文科学声明、有官方 qrels，适合第一周接 BEIR。

**子集策略（控制 API 成本）：**

1. 用 `datasets` 或 `beir` 加载 `BeIR/scifact`
2. **Corpus：** 取前 **500** 篇 `corpus` 文档索引（约 2–4 万 token，视 API 计费调整）
3. **Queries：** 取 **100** 条 dev/validation queries（或全量 300 若时间充裕）
4. **指标：** 官方 qrels 算 **Recall@5**、**nDCG@10**（MRR 可选）

**目录结构（Day 3 创建）：**

```text
tests/eval/beir_scifact/
  README.md              # 下载命令、子集大小、许可说明
  download.py            # 可选：缓存 corpus/queries/qrels 到本地 jsonl
  corpus_500.jsonl       # 生成后 gitignore（体积大）
  queries_100.jsonl
  qrels.jsonl
```

**与本地 gold 的区别：**

| | 本地 `gold_rag.jsonl` | BEIR scifact |
|--|----------------------|--------------|
| 语言 | 中文 | 英文 |
| 标注 | 关键词 / heading 启发式 | 官方 relevant doc id |
| 指标 | Recall@3（chunk 级启发式） | Recall@5（passage 级 qrels） |
| 目的 | 快速迭代、对齐 demo 文档 | 跨域泛化、对外展示 |

---

## 4. Agent E2E Eval 方案

### 4.1 两层断言（避免一上来测「答案质量」）

| 层 | 断言什么 | 是否需要 API |
|----|----------|--------------|
| **行为层** | 是否调用 `RAG_search_tool`、走 Self-RAG retry、CRAG trim | Mock LLM 即可 |
| **质量层** | 最终回答含 `expect_final_contains` | `requires_api` |

### 4.2 `gold_agent.jsonl` 字段

```json
{
  "id": "agent_001",
  "pattern": "react_full",
  "user_message": "在智能体优先的团队里，人类工程师的主要工作是什么？",
  "indexed_source": "get_start/工程技术：在智能体优先的世界中利用 Codex.md",
  "expect_tool": "RAG_search_tool",
  "expect_tool_min_calls": 1,
  "expect_no_tool": false,
  "expect_final_contains": ["工程师", "智能体"],
  "expect_metadata": {"self_rag_need_retrieve": true},
  "difficulty": "medium",
  "notes": "应先检索再作答"
}
```

**Case 分层（15+ 条）：**

| 类型 | 数量 | 测什么 |
|------|------|--------|
| 应检索 | 5 | `expect_tool = RAG_search_tool` |
| 不应检索 | 3 | 寒暄 / 纯计算 → `expect_no_tool: true` |
| Self-RAG retry | 3 | ungrounded → 二次检索（mock 或 API） |
| CRAG trim | 2 | 噪声 passage 被裁掉 |
| Feedback requery | 2 | 用户纠正 → requery metadata |

### 4.3 Mock 策略

- **LLM：** `AsyncMock`，按 `user_message` 返回预设 `AIMessage`（含或不含 `tool_calls`）
- **RAG：** 绑定 `tests/conftest.py` 的 `mock_embedder` + `in_memory_vector_store`，索引 `sample_markdown` 或 Codex 文档片段
- **真实 API 子集：** 标记 `@pytest.mark.requires_api`，每周跑 3–5 条代表性 case

---

## 5. 逐日计划（每天 8h，10 个工作日）

### Week 1 — 读懂测试 + RAG Eval 基础

#### Day 1 — 测试导读 A：基础设施 + 全量 RAG

| 时段 | 任务 | 产出 |
|------|------|------|
| 0.5h | 跑 `pytest -c tests/pytest.ini -m "unit or integration" -v`，确认 107 绿 | baseline 日志 |
| 0.5h | 读 `conftest.py` + `tests/fakes/vector_store.py` | 理解 MockEmbedder 策略 |
| 3h | 按 [TEST_CATALOG.md §1](./TEST_CATALOG.md) 读 `tests/rag/` 全部；**每文件跑一遍 pytest** | 笔记：S2B / chunker / pipeline 契约 |
| 2h | 精读 `test_pipeline_integration.py` + 对照 `rag/core.py` | 能口述 index→query 链路 |
| 1h | 精读 `test_parent_builder` + `test_small_to_big` 各选 1 个 test 单步调试 | 理解 parent window |
| 1h | 整理「我还没懂」清单 | 留给 Day 2 缓冲 |

**Day 1 退出标准：** 能解释「MockEmbedder 下 retriever 为何能命中」；笔记覆盖 rag/ 31 个 test。

---

#### Day 2 — 测试导读 B：Tools + Agent 骨架

| 时段 | 任务 | 产出 |
|------|------|------|
| 2h | [TEST_CATALOG.md §2](./TEST_CATALOG.md)：`test_rag_tool.py` + `test_tool_box.py` 全跑 | 理解 bind_rag_context |
| 2h | [TEST_CATALOG.md §3.1](./TEST_CATALOG.md)：`test_graph_and_nodes.py` | 理解 if_tool_calls / tool_node |
| 2h | 开始 `test_self_rag.py`（前半：rule + pre/post 节点） | Self-RAG metadata 清单 |
| 1h | 补 Day 1 遗留问题 | |
| 1h | 画一张 ReAct 主循环草图（llm → tools → llm） | 贴进个人笔记 |

**Day 2 退出标准：** 能说明 `RAG_search_tool` 从 ToolBox 到 store 的路径。

---

#### Day 3 — 测试导读 C：Self-RAG + CRAG + Feedback

| 时段 | 任务 | 产出 |
|------|------|------|
| 2h | `test_self_rag.py` 后半 + pattern build 测试 | Self-RAG 路由表 |
| 3h | `test_crag.py`（先 parsers → verdict → subgraph → wrapper） | CRAG 状态图 |
| 2h | `test_feedback.py` | feedback 与 self_rag 入口顺序 |
| 1h | 填写 TEST_CATALOG §5「改了 X 该跑谁」个人版 | 改代码速查表 |

**Day 3 退出标准：** 能区分 unit（测路由）和 eval（测质量）的边界；**测试导读阶段结束**。

---

#### Day 4 — RAG Eval：启用本地 gold

| 时段 | 任务 | 产出 |
|------|------|------|
| 0.5h | 确认 `.env`；`rag_demo.py --rag --in-memory` | API 可用 |
| 2h | 扩 `gold_rag.jsonl` → **20 条** | 含 difficulty / must_not_contain |
| 2.5h | 实现 `test_rag_gold_recall_at_3`，**去掉 skip** | eval 可跑 |
| 1h | 扩展 `chunk_matches()` + helper 单测 | |
| 1h | `eval_results.md` 记入 baseline Recall@3 | |
| 1h | 调阈值（建议 `mean Recall@3 >= 0.55`） | Day 4 交付 |

**Day 4 退出标准：** `pytest tests/eval/test_rag_metrics.py -m "eval and requires_api"` 通过。

---

#### Day 5 — Profile 对比 + Integration 补洞

| 时段 | 任务 | 产出 |
|------|------|------|
| 2h | `tests/eval/run_profiles.py`：4 profile × 本地 gold | JSON 结果 |
| 1h | 更新 `eval_results.md` 对比表 | |
| 2h | `test_pipeline_integration`：S2B 端到端 | 新用例 |
| 1.5h | HyDE mock integration | |
| 1h | gold 补 5 条难例 | ≥ 25 条 |
| 0.5h | 重跑 profile | Week 1 收束 |

**Week 1 退出标准：** 现有 108 test **全部能说出在保护什么**；本地 RAG eval + 4 profile 有数字。

---

### Week 2 — BEIR + Agent Eval + 汇总

#### Day 6 — BEIR scifact 接入

| 时段 | 任务 | 产出 |
|------|------|------|
| 1h | `pip install beir datasets`；`tests/eval/beir_scifact/README.md` | |
| 2h | `download.py` → corpus_500 / queries_100 / qrels | 本地 jsonl |
| 2.5h | 批量 `aindex` + 进度日志 | 可复现索引 |
| 2h | `test_beir_metrics.py`：Recall@5 / nDCG@10 | |
| 0.5h | baseline 首跑记入 eval_results | |

**Day 6 退出标准：** BEIR baseline Recall@5 可复现。

---

#### Day 7 — BEIR Profile + 难例分析

| 时段 | 任务 | 产出 |
|------|------|------|
| 2h | `run_profiles.py --dataset beir_scifact` 四 profile | BEIR 对比表 |
| 2h | 5 条失败 case study | eval_results 分析节 |
| 2h | 抽样 10 chunk 肉眼 review（英文科学文） | 分块笔记 |
| 1.5h | 可选 full + rerank（`@pytest.mark.slow`） | |
| 0.5h | 中文 gold vs BEIR 结论对比 | |

**Day 7 退出标准：** 知道哪个 profile 在 scifact 上有效。

---

#### Day 8 — Agent Eval：Mock E2E

| 时段 | 任务 | 产出 |
|------|------|------|
| 1h | `gold_agent.jsonl` schema + 15 条 | |
| 3h | `test_agent_metrics.py` 框架 + 行为层断言 | mock 全绿 |
| 2h | Self-RAG / CRAG / Feedback 各 2 条 mock case | 复用现有 test 模式 |
| 1h | 对照 TEST_CATALOG：新 eval 与旧 unit 不重复 | 去重说明 |
| 1h | `pytest tests/eval/test_agent_metrics.py -m "eval and not requires_api"` | |

**Day 8 退出标准：** 无 API 的 Agent e2e 可 CI 化。

---

#### Day 9 — Agent 真实 API + 联调

| 时段 | 任务 | 产出 |
|------|------|------|
| 2h | Codex 索引 + `bind_rag_context` 接 Agent | |
| 2h | 5 条 `requires_api` e2e | tool + 答案关键词 |
| 2h | `react` / `react_full` / `react_all` 对比 | pattern 表 |
| 1.5h | 修联调 bug | |
| 0.5h | 更新 eval_results Agent 节 | |

**Day 9 退出标准：** ≥ 3/5 API case 通过。

---

#### Day 10 — 汇总 + 文档 + 机动缓冲

| 时段 | 任务 | 产出 |
|------|------|------|
| 2h | 全量 eval 重跑 | 最终数字 |
| 1.5h | 完善 `eval_results.md` + 两周结论 | |
| 1.5h | 更新 `TESTING.md` / `TESTING_GUIDE.md` | |
| 1h | 消化 Week 1「还没懂」清单 | 或补读 test |
| 1h | 可选 GitHub Actions（仅 unit+integration） | |
| 1h | 第三周预览（HotpotQA、gold×50） | |

**Day 10 退出标准：** `eval_results.md` 可独立阅读；改 RAG/Agent 前知道跑哪些 test。

---

## 6. 每日例行（8h 内嵌）

```text
08:00–08:30  昨日 eval 数字回顾 + 今日目标
08:30–12:00  核心开发（测试 / eval 实现）
12:00–13:00  午休
13:00–17:00  跑实验 + 记结果 + 修失败 case
17:00–17:30  更新 eval_results.md + git commit（仅本地，按你习惯）
17:30–18:00  明日任务预习
```

**原则：** 先写断言、再跑实验；失败 case 先记入 `eval_results.md` 再决定改代码还是改 gold。

---

## 7. 风险与应对

| 风险 | 应对 |
|------|------|
| Embedding API 费用 / 限速 | BEIR 固定 500 corpus；索引加 cache；重复 eval 复用同一 collection |
| BEIR 全英文，与中文 gold 结论不一致 | 正常；分开记录，Week 2 再加 fiqa 或中文子集 |
| Agent API eval 不稳定 | 行为层以 mock 为准；API 层阈值宽松 + 少样本 |
| Reranker 下载慢 | 仅 `full` + `@pytest.mark.slow`，周末跑一次 |
| 启发式 `chunk_matches` 与人工感受不符 | Week 2 引入 `expected_chunk_id` 或 LLM-as-judge |

---

## 8. 本周不做的（明确边界）

- 不接 HotpotQA / MultiHop-RAG（下周）
- 不改 `rag/`、`agent/` 业务逻辑（除非 eval 暴露明确 bug）
- 不做 LLM-as-judge 全自动标注（成本高）
- 不 force push、不提交 `.env` / 大体积 BEIR 原始数据

---

## 9. 附录 A：`eval_results.md` 模板

创建 `docs/eval_results.md` 时可直接复制：

```markdown
# Eval Results

> 自动/手动评测记录。Profile 定义见 [WEEKLY_EVAL_PLAN.md](./WEEKLY_EVAL_PLAN.md)。

## 本地 Gold（Codex 中文文档）

| Date | Profile | Recall@3 | Cases | Notes |
|------|---------|----------|-------|-------|
| 2026-06-08 | baseline | — | 20 | 首日基线 |

## BEIR scifact（500 corpus / 100 queries）

| Date | Profile | Recall@5 | nDCG@10 | Notes |
|------|---------|----------|---------|-------|
| 2026-06-10 | baseline | — | — | 首跑 |

## Agent E2E

| Date | Pattern | Mock pass | API pass (n/N) | Notes |
|------|---------|-----------|----------------|-------|
| 2026-06-13 | react_full | — | —/5 | |

## 失败 Case 速记

- （日期）query / case id — 原因 — 待办

## 周结论

- （周日填写）
```

---

## 10. 附录 B：下周预览（不在本周范围）

1. BEIR `fiqa` 或 `trec-covid` 第二个子集  
2. HotpotQA distractor：多跳 + Agent 多轮 `rag_search`  
3. `gold_rag.jsonl` / `gold_agent.jsonl` 各扩到 50 条  
4. LLM-as-judge 抽样 20 条做答案质量相关性  
5. GitHub Actions 可选 `workflow_dispatch` 触发 eval

---

## 11. 快速索引

| 我想… | 去看 |
|-------|------|
| **某个 test 是什么意思** | [TEST_CATALOG.md](./TEST_CATALOG.md) |
| 跑日常回归 | [TESTING.md §3](./TESTING.md) |
| 写新 unit 测试 | [TESTING_GUIDE.md §2](./TESTING_GUIDE.md) |
| 扩展 gold 格式 | [TESTING_GUIDE.md §5](./TESTING_GUIDE.md) |
| 两周每天干什么 | 本文 §5 |
| 记录实验结果 | [eval_results.md](./eval_results.md) |
