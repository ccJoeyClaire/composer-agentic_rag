# 指标说明

[`legacy/_eval_`](../../legacy/_eval_/) 在 BEIR 格式数据集上输出的 **doc-level 排序指标**：各列含义、计算公式、读表速查。

← [报告索引](./README.md)

实现见 [`legacy/_eval_/scoring/metrics.py`](../../legacy/_eval_/scoring/metrics.py)；打分入口见 [`legacy/_eval_/rag_eval/pipeline.py`](../../legacy/_eval_/rag_eval/pipeline.py) 的 `score_queries`。

---

## 1. 评测在量什么

对每条 query，系统会：

1. 用 RAG retriever 拉取 `fetch_chunks = max(k_values) × chunk_fetch_multiplier` 个 **chunk**（默认 `20 × 4 = 80`）。
2. 按检索顺序 **去重到 document 级**，得到 `ranked: list[DocId]`（同一 doc 只保留排名最靠前的 chunk）。
3. 与 BEIR **qrels**（人工标注的相关性分数）对比，计算 g 个指标。
4. 对所有 query 的同一指标取 **算术平均**，得到结果 JSON 里的 `mean_metrics` 表头。

因此，表格里的数字不是「chunk 命中率」，而是 **「按 doc 排序后，前 k 个 doc 有多好」**。

### Gold 与相关性分数

| 概念 | 来源 | 用途 |
|------|------|------|
| **RelevanceMap** | 该 query 在 qrels 中的 `{doc_id: score}` | nDCG：使用**完整分级分数** |
| **gold** | `score >= rel_threshold` 的 doc 集合（默认 `rel_threshold=1`） | Recall / Hit / MRR |

BEIR 数据集的 score 是**分级**的（例如 trec-covid 为 0/1/2，msmarco 为 1）。`rel_threshold` 决定「算作 relevant」的门槛；可在 `rag_eval_arg_config.yaml` 的 `pool.rel_threshold` 调整。

### Pooled subset（索引范围）

完整 BEIR corpus 往往很大，实际只索引 **pool**：每个 query 的全部 gold doc + 最多 `max_distractors_per_query` 个标注为不相关的 distractor（默认 100）。Recall 在 pool 内是可达到的；指标衡量的是在 **有限候选集上的排序能力**，而非全库百万 doc 上的绝对 recall。

---

## 2. 指标列一览

默认 `k_values: [3, 10, 20]`，表头为：

`recall@3` · `ndcg@3` · `hit@3` · `recall@10` · `ndcg@10` · `hit@10` · `recall@20` · `ndcg@20` · `hit@20` · `mrr@20`

| 列 | 类型 | 一句话 |
|----|------|--------|
| **Recall@k** | 连续 [0, 1] | 前 k 个 doc 里找到了多少比例的 gold doc |
| **nDCG@k** | 连续 [0, 1] | 前 k 的排序质量（含分级相关性 + 位置折扣） |
| **Hit@k** | 二值 per-query，再平均 | 有多少比例的 query 在前 k 里**至少**命中 1 个 gold |
| **MRR@20** | 连续 [0, 1] | 第一个 gold doc 出现得有多靠前（cutoff = 20） |

**读表习惯**（与 `hotpotqa.ipynb` 分析一致）：

- 主指标：**nDCG@10** — 兼顾相关性与排序。
- 辅助：**MRR@20**（首个相关 doc 多快出现）、**Recall@10**（多 gold 场景下的覆盖率）。

---

## 3. 各指标详解

### 3.1 Recall@k

**公式（单 query）**

```
hits = |{ doc ∈ ranked[:k] : doc ∈ gold }|
Recall@k = hits / |gold|     （|gold| = 0 时记 0）
```

**计算示例**：gold = {A, B}（2 个），ranked 前 10 = [C, A, D, B, …]  
→ hits = 2 → **Recall@10 = 1.0**；前 3 只有 A → **Recall@3 = 0.5**。

**深层含义**

- 衡量 **覆盖率**：需要找齐所有相关文档时（如 HotpotQA 多跳、需多篇证据），Recall 比 Hit/MRR 更严格。
- 对 `num_gold > 1` 的 query 天然更难打满；对比 profile 时可结合 per-query 的 `num_gold` 诊断（见分析 notebook）。
- k 越大，Recall 单调不降（只看前 k，不会「删掉」已找到的 doc）。

---

### 3.2 Hit@k（Success@k）

**公式（单 query）**

```
Hit@k = 1.0  若 ∃ doc ∈ ranked[:k] 使得 doc ∈ gold
        0.0  否则
```

报告中的 **Hit@k** = 所有 query 的 Hit@k 的 **平均值**（即 **Success Rate@k**）。

**与 Recall 的区别**

| 场景 | Recall@k | Hit@k |
|------|----------|-------|
| gold = {A, B}，前 k 只命中 A | 0.5 | **1.0** |
| gold = {A}，前 k 命中 A | 1.0 | 1.0 |
| gold = {A}，前 k 未命中 | 0.0 | 0.0 |

Hit 只问「有没有至少一个 relevant」；多 gold query 上 Hit 往往 **高于** Recall，更乐观。

**深层含义**

- 近似「这条 query 检索是否 **可用**」：RAG 只要有一篇好 doc 就能答一点。
- **Hit@3** 对应极短 context（如前 3 个 doc/chunk）；**Hit@10** 接近常见 `top_k=10` 设定。

---

### 3.3 MRR@k（Mean Reciprocal Rank）

**公式（单 query，cutoff = k）**

```
若在 ranked[0..k-1] 中第一个 gold 出现在 rank r（1-based）：
    MRR@k = 1/r
否则：
    MRR@k = 0
```

报告列 **mrr@20** 使用 `k = max(k_values) = 20`（见 `RunConfig.max_k`）。

**计算示例**：ranked = [X, Y, A, …]，A ∈ gold，r = 3 → **MRR = 1/3 ≈ 0.333**。

**深层含义**

- 只关心 **第一个** relevant doc 的位置，不关心其余 gold 或分级高低。
- 排名从第 1 提到第 2，MRR 从 1.0 跳到 0.5，对 **头部排序** 非常敏感。
- 适合「用户看第一条结果就够」的场景；对「必须凑齐多篇证据」的问题，Recall/nDCG 信息量更大。

**MRR vs nDCG**：MRR 是二值 relevant + 首个位置；nDCG 用分级分数且累加前 k 所有位置。

---

### 3.4 nDCG@k（Normalized Discounted Cumulative Gain）

**公式**

对前 k 个 ranked doc，取 qrels 中的 gain（未标注则 0）：

```
DCG@k = Σ_{i=1..k}  gain_i / log₂(i + 1)

ideal_gains = qrels 中所有 score 降序取前 k
IDCG@k = DCG(ideal_gains)

nDCG@k = DCG@k / IDCG@k    （IDCG = 0 时记 0）
```

位置 1 的分母为 log₂(2) = 1；越靠后，同一 gain 的贡献越小（**位置折扣**）。

**计算直觉**

- 把 **高相关 doc 排在前面** → nDCG 高。
- 只 relevant 但顺序乱 → nDCG 低于理想排序。
- 用了 **全部分级 score**（不限于 gold 阈值），例如 trec-covid 中 score=2 比 score=1 贡献更大。

**深层含义**

- 信息检索里 **排序质量** 的标准指标：同时看「找到了谁」和「排得对不对」。
- 对 RAG ablation，**nDCG@10** 通常是 profile 对比的主轴（reranker、HyDE、chunk 策略等改的是排序，而不只是有没有命中）。
- **nDCG@3 vs @10 vs @20**：k 小 → 只看最顶部；k 大 → 深位次也有贡献，但分母 IDCG 也变大，需在同一 k 下横向比。

---

## 4. 指标之间的关系（读表速查）

```
                    关心什么？
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    「有没有？」      「找齐了吗？」    「排得好吗？」
      Hit@k           Recall@k          nDCG@k
         │               │               │
         └─────── MRR@k：第一个 relevant 多靠前 ───────┘
```

| 若你关心… | 优先看 |
|-----------|--------|
| RAG 能否拿到任意可用证据 | Hit@3 / Hit@10 |
| 多文档 / 多跳是否找全 | Recall@10 / Recall@20 |
| 第一条结果是否立刻 relevant | MRR@20 |
| 整体排序与分级相关性 | **nDCG@10** |

同一 profile 下通常：**Hit@k ≥ Recall@k**（单 query，多 gold 时严格不等）；k 越大，Recall / Hit / nDCG 一般 **不降**（MRR@k 随 k 增大可能升高，因为 cutoff 内更容易出现 gold）。

---

## 5. 聚合与结果文件

- **Per-query**：`results[].per_query[].metrics` 含上述各键。
- **Profile 均值**：`mean_metrics` = 对所有 query 的同名指标 **简单算术平均**（[`mean_metrics`](../../legacy/_eval_/scoring/metrics.py)）。
- 未纳入评分的 query：`queries_with_gold` 过滤掉 **没有任何 gold** 的 query（否则 Recall 分母为 0）。

对比两个 profile 时，分析模块用 **Δ = challenger − reference**（正数表示 challenger 更好），并支持 winrate、paired t-test 等（见 `legacy/_eval_/analysis/`）。

---

## 6. 写 BEIR 结果报告时的建议结构

1. **实验设置**：dataset、query 数量、`rel_threshold`、pool 大小（`num_docs_indexed`）、profiles、`k_values`。
2. **主表**：各 profile 的 `mean_metrics`（至少 nDCG@10、MRR@20、Recall@10）。
3. **解读**：
   - 哪个 ablation 对 **排序**（nDCG/MRR）帮助最大；
   - 哪个对 **覆盖率**（Recall）帮助最大；
   - Hit@3 vs Hit@10 是否说明改进主要在 **极短 context** 还是 **更深检索**。
4. **局限**（建议在报告中写明）：
   - Doc-level 去重：同一 doc 多个 chunk 只计一次，偏 favor「先命中该 doc 的 chunk 策略」；
   - Pooled subset：非全库检索；
   - 指标不衡量 **生成答案** 是否正确（端到端见 RAGChecker / `docs/Eval_report.md`）。

---

## 7. 代码索引

```python
# 单 query 指标集合（legacy/_eval_/scoring/metrics.py）
for k in k_values:
    scores[f"recall@{k}"] = recall_at_k(ranked, gold, k)
    scores[f"ndcg@{k}"]   = ndcg_at_k(ranked, relevance, k)
    scores[f"hit@{k}"]    = hit_at_k(ranked, gold, k)
scores[f"mrr@{mrr_k}"] = mrr_at_k(ranked, gold, mrr_k)  # mrr_k = max(k_values)
```

```python
# ranked 列表来源（legacy/_eval_/rag_eval/pipeline.py）
chunks = await retriever.aquery(query.text, top_k=cfg.fetch_chunks)
ranked = ranked_doc_ids(chunks)  # chunk → doc，保序去重
gold   = gold_docs(qrels[qid], cfg.pool_spec.rel_threshold)
```

