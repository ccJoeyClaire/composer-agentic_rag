# 测试用例解读目录

> **读者：** 已跑过 `pytest` 但想搞懂「每个 test 在保护什么」的开发者。  
> **用法：** 按模块顺序阅读；每读完一节，单独跑该文件并对照断言。  
> **配套：** [TESTING.md](./TESTING.md) · [WEEKLY_EVAL_PLAN.md](./WEEKLY_EVAL_PLAN.md)

**统计：** 共 **112** 个 test function（其中 1 个 eval 主用例当前 `@pytest.mark.skip`）。

---

## 0. 先搞懂的三件事

### 0.1 测试在回答什么问题？

| 问题 | 哪类 test 回答 |
|------|----------------|
| 函数算得对不对？ | `unit` — 如 `merge_windows`、`rule_based_need_retrieve` |
| 组件接在一起能跑吗？ | `integration` — 如 index → store → query |
| 检索/Agent 好不好？ | `eval` — `gold_*.jsonl` + Recall@k（本周要做） |

### 0.2 公共基础设施（读测试前先看）

| 文件 | 作用 |
|------|------|
| `tests/conftest.py` | `MockEmbedder`（同文本→同向量）、`sample_markdown`、`make_chunk` 工厂 |
| `tests/fakes/vector_store.py` | 轻量内存向量库，integration 不依赖 Docker |
| `tests/pytest.ini` | marker 定义、`pythonpath = ..` |

**`MockEmbedder` 的含义：** 不是真 embedding。故意让「query 文本 = chunk 文本」时相似度最高，这样 integration 可以断言「命中了哪条」而不调 API。

### 0.3 建议的学习顺序（约 12–14h，有本文档）

```text
Day A（4h）  conftest → rag/ 全部
Day B（4h）  tools/ 全部 → legacy_agent/test_graph_and_nodes.py
Day C（4h）  legacy_agent/test_self_rag → test_crag → test_feedback
Day D（2h）  agent/（新版）+ eval/ + 回头补看不懂的 test
```

单独跑某一文件：

```bash
pytest -c tests/pytest.ini tests/rag/test_parent_builder.py -v
pytest -c tests/pytest.ini tests/legacy_agent/test_crag.py::test_decide_action -v
pytest -c tests/pytest.ini tests/agent/test_builder.py -v
```

---

## 1. RAG 模块（35 个 test）

### 1.1 `tests/rag/test_parent_builder.py`（8 + 1 debug）— Small-to-Big 索引侧

**对应生产代码：** `rag/document_augmentation/parent_builder.py`  
**核心概念：** 把小 chunk 挂上 `chunk_id`、`anchor_window`（父窗口成员列表），检索命中小 chunk 后再拼成大段。

| Test | 在保护什么 | 读法提示 |
|------|------------|----------|
| `test_assign_parent_chunks_empty` | 空输入不崩 | 边界 case |
| `test_assign_parent_chunks_metadata_contract` | 每个小 chunk 有稳定 id（`kb.md::1`）、section_id、window | **最重要**，理解 S2B 索引结构 |
| `test_assign_parent_chunks_skips_windows_when_budget_disabled` | `parent_token_budget=0` 时只写 id，不写 window | 关闭 S2B 时的降级路径 |
| `test_assign_parent_chunks_respects_section_boundaries` | 不同 `heading_path` 不跨章节合并 parent | 防止父窗口串台 |
| `test_assign_parent_chunks_token_budget_limits_window` | 索引阶段 window 扩展受 token 预算约束 | anchor 居中扩展，邻居装不下就丢弃 |
| `test_windows_overlap_and_merge` | 两个 anchor window 重叠时能 merge | 检索多 hit 合并逻辑的基础 |
| `test_cluster_overlapping_hits_transitive_closure` | A 与 B 重叠、B 与 C 重叠 → A,B,C 同一簇 | 传递闭包，避免漏合并 |
| `test_materialize_parent_content_strips_char_overlap` | 拼 parent 正文时去掉相邻 chunk 字符重叠 | 防止 duplicated text |
| `test_debug_assign_parent_chunks_output` | 打印索引后 metadata（需 `DEBUG_PROBES=1`） | 手工探针，默认 skip |

**⏱ 建议：** 45–60 min（先读 `assign_parent_chunks` 源码再读 test）

---

### 1.2 `tests/rag/test_small_to_big.py`（8）— Small-to-Big 检索侧

**对应生产代码：** `rag/retriever/small_to_big_retriever.py`  
**核心概念：** 向量命中 small chunk → 聚类 → 物化 parent；预算与 `top_k` 并列，超预算时**减少 parent 数量**，单个 parent **完整物化** merged window，不做内容截断。

| Test | 在保护什么 |
|------|------------|
| `test_expand_single_hit_materializes_parent` | 单个小 chunk 命中 → 展开成 parent 内容 |
| `test_expand_overlapping_hits_merge_to_one_parent` | 多个命中 window 重叠 → 只返回一个 parent |
| `test_expand_budget_limits_parent_count_over_top_k` | 总预算 = `budget × top_k` 时少返回 parent，不截断正文 | **最重要**，理解查询侧预算策略 |
| `test_expand_merged_cluster_materializes_full_window` | 多 hit 合并后完整物化 window（含首尾 hit） | 合并后不丢 hit、不裁 chunk |
| `test_expand_disjoint_hits_return_multiple_parents` | 不重叠 → 多个 parent |
| `test_expand_fetches_missing_members_from_store` | window 里有的 member 不在 hit 列表 → 从 store 回查 |
| `test_small_to_big_retriever_empty_query` | 空 query 返回空 |
| `test_small_to_big_retriever_recall_multiplier` | 内部 recall 放大（先多捞小 chunk 再扩 parent） |

**⏱ 建议：** 40–50 min

---

### 1.3 `tests/rag/test_semantic_chunker.py`（3）

**对应生产代码：** `rag/chunker/semantic_chunker.py`

| Test | 在保护什么 |
|------|------------|
| `test_semantic_chunker_extracts_heading_path` | Markdown 标题进入 `metadata.heading_path` |
| `test_semantic_chunker_respects_token_limit` | 不超过 `chunk_tokens` 上限 |
| `test_semantic_chunker_semantic_break_increases_chunk_count` | 语义切分比硬切产生更多边界 |

**⏱ 建议：** 25 min

---

### 1.4 `tests/rag/test_contextual_enricher.py`（4）

**对应生产代码：** `rag/document_augmentation/context_enricher.py`

| Test | 在保护什么 |
|------|------------|
| `test_build_contextual_header_includes_doc_and_section` | 索引 header 含文档名和章节 |
| `test_build_embed_text_joins_header_and_body` | 真正送去 embed 的是 header+正文 |
| `test_aenrich_for_index_sets_embed_text` | 索引后 metadata 有 `embed_text` |
| `test_aenrich_chunks_prepends_header_on_retrieve` | 检索展示时 prepend header（给人看） |

**⏱ 建议：** 30 min

---

### 1.5 `tests/rag/test_predict_question.py`（8）

**对应生产代码：** `rag/document_augmentation/predict_question.py`

| Test | 在保护什么 |
|------|------------|
| `test_parse_predicted_questions_from_json` | LLM 返回 JSON 能解析出问题列表 |
| `test_parse_predicted_questions_invalid_json` | 坏 JSON 不崩，降级为空 |
| `test_append_questions_to_embed_text` | 预测问题追加进 embed 文本 |
| `test_predict_questions_for_chunk_calls_llm` | 单 chunk 会调 LLM（mock） |
| `test_augment_chunks_runs_llm_calls_concurrently` | 多 chunk 并发调 LLM |
| `test_bounded_concurrency_caps_at_499` | 并发上限 499（防打爆 API） |
| `test_augment_chunks_with_questions_sets_metadata_and_embed_text` | metadata 和 embed 都更新 |
| `test_predict_question_enricher_aenrich_for_index` | enricher 接口与 indexer 对接 |

**⏱ 建议：** 45 min（注意这里大量 `AsyncMock`）

---

### 1.6 `tests/rag/test_pipeline_integration.py`（3）— 最重要的一条链路

**对应生产代码：** `rag/core.py` 的 `RAGIndexer` + `RAGRetriever`

| Test | 在保护什么 |
|------|------------|
| `test_indexer_stores_chunks` | `aindex` 后 store 里真有 chunk |
| `test_retriever_returns_matching_chunk` | query 与某 chunk 文本一致时能命中（靠 MockEmbedder） |
| `test_retriever_applies_top_k_without_reranker` | 返回条数 ≤ `top_k` |

**⏱ 建议：** 30 min — **理解 MockEmbedder 策略的最佳入口**

---

## 2. Tools 模块（26 个 test）

### 2.1 `tests/tools/test_tool_box.py`（6）

**对应生产代码：** `tools/tool_box.py`、`tools/registry.py`

| Test | 在保护什么 |
|------|------------|
| `test_resolve_decorated_tool` | 按路径解析 `@local_tool` 函数 |
| `test_autodiscover_registers_decorated_tools` | 自动扫描包注册工具 |
| `test_ainvoke_integrate` | `ainvoke` 能调用并返回 `ToolResult` |
| `test_list_tools_schema` | 列出工具时带 JSON schema |
| `test_local_tool_decorator_sets_metadata` | 装饰器写入 `TOOL_INFO_ATTR` |
| `test_autodiscover_can_be_disabled` | 可关掉自动发现 |

**⏱ 建议：** 30 min

---

### 2.2 `tests/tools/test_decorators.py`（4）

| Test | 在保护什么 |
|------|------------|
| `test_local_tool_attaches_tool_info` | 装饰后函数仍可调用，且带元数据 |
| `test_integrate_function_still_callable` | 装饰不破坏原函数 |
| `test_local_tool_appears_in_isolated_tool_box_schema` | 隔离 ToolBox 能看到 schema |
| `test_local_tool_invoked_via_isolated_tool_box` | 通过 ToolBox 调用 math_tool |

**⏱ 建议：** 20 min

---

### 2.3 `tests/tools/test_rag_tool.py`（11）— Agent 调 RAG 的桥梁

**对应生产代码：** `tools/LocalTool/RAG_tool.py`

| Test | 在保护什么 |
|------|------------|
| `test_rag_tools_have_decorator_metadata` | `RAG_index_tool` / `RAG_search_tool` 注册信息正确 |
| `test_rag_tools_appear_in_isolated_tool_box` | ToolBox 能发现 RAG 工具 |
| `test_rag_index_tool_requires_bind` | 未 `bind` 时报错提示 |
| `test_rag_search_tool_requires_bind` | 同上 |
| `test_rag_index_tool_indexes_document` | bind 后能索引 |
| `test_rag_search_tool_returns_indexed_content` | 索引后能搜到 |
| `test_legacy_fixed_retriever_notes_on_options` | 旧版 pinned retriever 时，hyde 等选项只记 note 不报错 |
| `test_full_context_caches_and_clamps_retriever_variants` | `bind_rag_context` 按 hyde/rerank/recall_n 缓存 retriever |
| `test_disabled_query_option_emits_note` | `allow_hyde=False` 时降级并说明 |
| `test_disabled_predict_questions_emits_note` | 索引侧选项被禁用时说明 |
| `test_rag_tools_via_tool_box` | 完整走 ToolBox.ainvoke 路径 |

**⏱ 建议：** 50 min — **写 Agent eval 前必读**

---

### 2.4 `tests/tools/test_mcp_tools.py`（5）

**对应生产代码：** `tools/MCPTool/`

| Test | 在保护什么 |
|------|------------|
| `test_mcp_registry_paths` | MCP 工具注册路径 |
| `test_mcp_tools_schema` | schema 结构 |
| `test_bocha_without_api_key` | 缺搜索 API key 时友好报错 |
| `test_convert_document_without_markitdown` | 缺 markitdown 时友好报错 |
| `test_convert_with_ocr_without_llm_key` | 缺 LLM key 时友好报错 |

**⏱ 建议：** 20 min（测的是「缺依赖时的错误信息」，不测真实 MCP 调用）

---

## 3. Agent 模块

新版实现在 `agent/`（capabilities 插件化）；旧版 CRAG / Self-RAG 在 `legacy/agent/`，测试在 `tests/legacy_agent/`。

### 3.0 `tests/agent/`（39）— 新版 LangGraph + capabilities

**对应生产代码：** `agent/builder.py`、`agent/capabilities/`、`agent/core/`

| 文件 | 在保护什么 |
|------|------------|
| `test_builder.py` | 图编译、LLM/tools 后路由、pattern 开关 |
| `test_retrieval_gate_node.py` | Retrieval Gate 打分与 verdict |
| `test_rag_profile_router_node.py` | RAG profile 校验与 metadata |
| `test_rag_tool_policy.py` | RAG tool query-only vs override schema |
| `test_output.py` | `OutputState` 序列化 |
| `test_pattern_config.py` | pattern yaml 加载 |

**⏱ 建议：** 改 `agent/` 下任何代码前，先 `pytest tests/agent/ -v`

---

### 3.1 `tests/legacy_agent/test_graph_and_nodes.py`（6）— 旧版 ReAct 骨架

**对应生产代码：** `legacy/agent/graph.py`（`if_tool_calls`）、`legacy/agent/nodes.py`

| Test | 在保护什么 |
|------|------------|
| `test_if_tool_calls_routes_to_tools` | AIMessage 带 `tool_calls` → 路由到 `tools` 节点 |
| `test_if_tool_calls_routes_to_end_when_no_tools` | 无 tool_calls → 不进 tools |
| `test_if_tool_calls_routes_to_end_for_human_message` | 最后一条是 HumanMessage → 不进 tools |
| `test_tool_node_returns_tool_messages` | `tool_node` 把工具输出包成 `ToolMessage` |
| `test_tool_node_surfaces_errors` | 工具抛错时错误信息进 ToolMessage |
| `test_llm_node_passes_tools_to_client` | `llm_node` 把 ToolBox schema 传给 LLM |

**⏱ 建议：** 40 min — **理解 LangGraph 主循环的第一步**

---

### 3.2 `tests/legacy_agent/test_self_rag.py`（14）— 检索前/后反思

**对应生产代码：** `legacy/agent/reflection/self_rag.py`、`legacy/agent/graph.py`

| Test | 在保护什么 |
|------|------------|
| `test_rule_based_need_retrieve` | 规则：像问题的句子 → 需要检索；寒暄 → 不需要 |
| `test_self_rag_pre_sets_need_retrieve` | pre 节点给 metadata 打上 `self_rag_need_retrieve` |
| `test_self_rag_pre_skips_without_human_message` | 最后不是用户消息 → 跳过 |
| `test_self_rag_post_skips_without_context` | 没有检索上下文 → post 跳过 |
| `test_self_rag_post_marks_grounded` | 有依据 → `self_rag_grounded=True` |
| `test_self_rag_post_adds_retry_hint_when_ungrounded` | 无依据 → 允许 retry + hint |
| `test_self_rag_pre_node_runs_with_config` | `SelfRagConfig` 注入生效 |
| `test_self_rag_post_node_runs_with_config` | 同上 |
| `test_if_after_llm_routes_to_self_rag_post` | LLM 回复后 → 进 self_rag_post |
| `test_if_after_llm_routes_tool_calls_to_tools` | 有 tool_calls → 仍进 tools |
| `test_route_after_self_rag_post_retries_when_ungrounded` | 未 grounded 且允许 retry → 回 llm |
| `test_route_after_self_rag_post_ends_when_grounded` | grounded → END |
| `test_build_agent_self_rag_pattern` | `react_self_rag` 图结构包含 pre/post 节点 |
| `test_build_agent_full_pattern_has_crag_and_self_rag` | `react_full` 同时含 CRAG + Self-RAG |

**⏱ 建议：** 60–75 min

---

### 3.3 `tests/legacy_agent/test_crag.py`（14）— 检索结果校正

**对应生产代码：** `legacy/agent/subgraph/CRAG.py`、`legacy/agent/reflection/parsers.py`

**分层理解：**

```text
parsers（拆分 RAG 工具输出）
  → compute_verdict / decide_action（纯函数策略）
    → build_crag_subgraph（独立子图，私有 CragState）
      → build_crag_node（挂到主 Agent 的 wrapper）
```

| Test | 在保护什么 |
|------|------------|
| `test_split_rag_chunks` | ToolMessage 里 `\n\n---\n\n` 拆成 passage 列表 |
| `test_extract_rag_tool_results` | 从 messages 抽出 RAG 查询和原始命中 |
| `test_compute_verdict` | 打分标签聚合为 correct / incorrect / ambiguous |
| `test_decide_action` | 根据 verdict + 尝试次数决定 use / requery / degrade / web_fallback |
| `test_subgraph_use_when_all_correct` | 全 correct → 直接用全部 passage |
| `test_subgraph_trims_to_correct_passages` | 混合标签 → 只保留 correct |
| `test_subgraph_reretrieves_then_succeeds` | incorrect → requery 后成功 |
| `test_subgraph_degrades_when_exhausted` | 次数用尽 → degrade（降级回答） |
| `test_subgraph_web_fallback_when_enabled` | 启用 web 且用尽 → web_fallback |
| `test_wrapper_skips_without_rag_tool` | 消息里没有 RAG 工具结果 → CRAG 节点跳过 |
| `test_wrapper_trims_and_rewrites_tool_message` | wrapper 把裁剪后的 context 写回 ToolMessage |
| `test_route_after_crag_always_returns_llm` | CRAG 后总是回 LLM 生成 |
| `test_build_react_agent_with_crag_has_crag_node` | 图里有 crag 节点 |
| `test_build_agent_react_crag_pattern` | `react_crag` pattern 可编译 |

**⏱ 建议：** 75–90 min — **最复杂的一组，建议画状态图**

---

### 3.4 `tests/legacy_agent/test_feedback.py`（15）— 用户纠正 / 澄清

**对应生产代码：** `legacy/agent/reflection/feedback.py`

| Test | 在保护什么 |
|------|------------|
| `test_default_detect_feedback_correction` | LLM 判定 correction |
| `test_default_detect_feedback_clarify` | LLM 判定 clarify |
| `test_default_detect_feedback_normal_question` | LLM 判定普通问题 → 非 feedback |
| `test_default_plan_requery_after_retrieval` | LLM plan：correction + 已有检索 → requery |
| `test_default_plan_clarify` | LLM plan：clarify |
| `test_detect_feedback_skips_without_llm` | 无 LLM 时不做 detect |
| `test_detect_feedback_clears_when_last_message_not_human` | 最后非用户消息 → 清 feedback 标记 |
| `test_detect_feedback_marks_correction` | detect 节点写入 metadata |
| `test_plan_feedback_sets_requery_metadata` | plan 节点写入 requery 计划 |
| `test_route_after_detect` | detect 后路由到 plan 或跳过 |
| `test_feedback_nodes_end_to_end` | detect → plan 串联 |
| `test_feedback_detect_clears_on_normal_question` | 普通问题不触发 feedback 流 |
| `test_build_agent_feedback_pattern` | `react_feedback` 图结构 |
| `test_build_agent_all_pattern` | `react_all` 含 feedback + crag + self_rag |
| `test_build_react_agent_feedback_entry_before_self_rag` | feedback 入口在 self_rag 之前 |

**⏱ 建议：** 60 min

---

## 4. Eval 模块（3 个 test）

### 4.1 `tests/eval/test_rag_metrics.py`

| Test | 在保护什么 | 状态 |
|------|------------|------|
| `test_recall_at_k_helper_detects_keyword_hit` | `recall_at_k` 启发式：关键词在 top_k 内算命中 | ✅ 常跑 |
| `test_load_gold_cases_reads_jsonl` | gold 文件格式可读 | ✅ 常跑 |
| `test_rag_gold_recall_at_3` | 对 Codex 文档跑真检索 eval | ⏸ **skip**，本周要启用 |

**⏱ 建议：** 15 min

---

## 5. 按「保护边界」速查

| 如果你改了… | 必须先绿的 test 文件 |
|-------------|---------------------|
| `parent_builder.py` | `test_parent_builder.py`, `test_small_to_big.py` |
| `semantic_chunker.py` | `test_semantic_chunker.py` |
| `context_enricher.py` | `test_contextual_enricher.py` |
| `predict_question.py` | `test_predict_question.py` |
| `core.py` indexer/retriever | `test_pipeline_integration.py` |
| `RAG_tool.py` | `test_rag_tool.py` |
| `agent/builder.py` / capabilities | `tests/agent/test_builder.py` 等 |
| `legacy/agent/graph.py` / `nodes.py` | `tests/legacy_agent/test_graph_and_nodes.py` + 相关 pattern test |
| `legacy/agent/self_rag.py` | `tests/legacy_agent/test_self_rag.py` |
| `legacy/agent/CRAG.py` | `tests/legacy_agent/test_crag.py` |
| `legacy/agent/feedback.py` | `tests/legacy_agent/test_feedback.py` |

---

## 6. 读 test 时的推荐笔记模板

每看完一个文件，在笔记里写一行：

```text
文件: test_crag.py
保护的核心契约: incorrect 且未用尽 → requery；用尽 → degrade 或 web
我还没懂: decide_action 和 subgraph 里 attempt 计数谁负责递增？
下次改代码前必跑: pytest tests/legacy_agent/test_crag.py -v
```

---

## 7. 与 Eval 的关系

| 现有 unit/integration | 不能替代 |
|----------------------|----------|
| 证明路由、metadata、chunk 结构 | 真实 embedding 下的 Recall@k |
| Mock LLM 的 CRAG 分支 | 真实 LLM 是否乱调工具 |
| MockEmbedder 命中 | 中文/英文文档上的检索质量 |

**所以本周路线：** 先用 1.5–2 天读懂现有 test（知道契约）→ 再写 eval（测质量）。
