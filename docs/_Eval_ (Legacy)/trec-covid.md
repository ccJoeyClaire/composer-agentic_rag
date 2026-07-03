# TREC-COVID — 检索 ablation

← [报告索引](./README.md) · [指标说明](./metrics.md)

配置：[`legacy/_eval_/configs/rag_eval_arg_config_trec-covid.yaml`](../../legacy/_eval_/configs/rag_eval_arg_config_trec-covid.yaml)

（暂无专用 visual notebook；可复用 `hotpotqa.ipynb` 并改 `RUN_JSON`。）

---

## 实验设置

| 项 | 值 |
|----|-----|
| 结果文件 | `legacy/_eval_/results/trec-covid_<timestamp>.json` |
| Queries | |
| `k_values` | 3, 10, 20 |
| `rel_threshold` | 1 或 2（trec-covid qrels 为 0/1/2） |
| `max_distractors_per_query` | 100 |
| 参考 profile | `baseline` |

---

## 1. Mean metrics（绝对值）

| profile | recall@3 | ndcg@3 | hit@3 | recall@10 | ndcg@10 | hit@10 | recall@20 | ndcg@20 | hit@20 | mrr@20 | docs |
|---------|----------|--------|-------|-----------|---------|--------|-----------|---------|--------|--------|------|
| token | | | | | | | | | | | |
| semantic | | | | | | | | | | | |
| semantic_rerank | | | | | | | | | | | |
| baseline | | | | | | | | | | | |
| baseline_s2b | | | | | | | | | | | |
| baseline_hyde | | | | | | | | | | | |
| baseline_s2b_hyde | | | | | | | | | | | |
| baseline_predict_q | | | | | | | | | | | |
| full | | | | | | | | | | | |

---

## 2. Δ heatmap vs reference profiles

![Δ heatmap](./assets/trec-covid/02_delta_heatmap.png)

---

## 3. Winrate vs `baseline`

![Winrate vs baseline](./assets/trec-covid/03_winrate_baseline.png)

---

## 4. Ablation ladder

![Ablation ladder](./assets/trec-covid/04_ablation_ladder.png)

---

## 5. Per-query Δ distribution

![Per-query delta](./assets/trec-covid/05_per_query_delta.png)

---

## 6. Paired t-test（`baseline` ablations）

![Paired t-test](./assets/trec-covid/06_paired_ttest.png)

---

## 7. Diagnostic: `num_gold` vs recall

![num_gold vs recall](./assets/trec-covid/07_num_gold_recall.png)

---

## 8. 解读（待写）

- [ ] …
