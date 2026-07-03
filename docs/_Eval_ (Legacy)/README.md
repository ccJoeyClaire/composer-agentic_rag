# Legacy BEIR 检索评测报告

`legacy/_eval_` 在 BEIR 格式数据集上的 **doc-level 检索 ablation** 结果归档。分析 notebook 在 [`legacy/_eval_/analysis/visual/`](../../legacy/_eval_/analysis/visual/)。

## 文档结构

| 文件 | 内容 |
|------|------|
| [metrics.md](./metrics.md) | 指标定义、公式、读表速查（与具体数据集无关） |
| [hotpotqa.md](./hotpotqa.md) | HotpotQA 跑分 + 图表 |
| [nfcorpus.md](./nfcorpus.md) | NFCorpus 跑分 + 图表（待填） |
| [msmarco.md](./msmarco.md) | MS MARCO 跑分 + 图表（待填） |
| [trec-covid.md](./trec-covid.md) | TREC-COVID 跑分 + 图表（待填） |

## 如何更新某一数据集

1. 跑 eval：`python -m _eval_.rag_eval.run --config _eval_/configs/rag_eval_arg_config_<dataset>.yaml`
2. 打开对应 notebook（如 `hotpotqa.ipynb`），确认 `RUN_JSON` 指向最新结果。
3. 依次运行各 section，把图 **另存为 PNG** 到 `assets/<dataset>/`（文件名见各数据集 md 里的占位）。
4. 从 notebook 的 mean metrics 表复制数字到 md 的表格；在 **解读** 小节写 3–5 条结论。

## 图片目录

```
docs/(Eval Legacy)/assets/
  hotpotqa/
  nfcorpus/
  msmarco/
  trec-covid/
```

Δ 符号约定（与 notebook 一致）：**challenger − reference**，正数表示 challenger 更好。

## 代码入口

- 指标：[`legacy/_eval_/scoring/metrics.py`](../../legacy/_eval_/scoring/metrics.py)
- 打分：[`legacy/_eval_/rag_eval/pipeline.py`](../../legacy/_eval_/rag_eval/pipeline.py) → `score_queries`
- 结果 JSON：`legacy/_eval_/results/<dataset>_<timestamp>.json`
