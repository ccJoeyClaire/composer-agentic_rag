# eval/（已迁移）

端到端评测 harness 已独立到 sibling 项目 **[composer-eval](../composer-eval)**。

- 源码：`composer-eval/eval/`
- 数据：`composer-eval/eval/data/`
- 运行：`python -m eval.harness.main`（在 composer-eval 目录下）

本目录下的 `data/` 为历史归档，新跑分请使用 composer-eval 中的副本。

RAG 库仍通过 `[project.optional-dependencies] eval` 提供 ragchecker 等依赖，但不再 ship `eval` Python 包。
