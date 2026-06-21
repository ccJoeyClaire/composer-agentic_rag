# QA Eval — script skeleton (Easy Dataset → profile ablation)

Post-build **custom long-document QA datasets** for proving RAG profile value via **answer quality**, not BEIR-style retrieval alone.

Context: NFCorpus-style benchmarks under-measure `use_contextual`, `use_small_to_big`, `use_hyde`, `use_predict_questions`. This pipeline ingests Easy Dataset (or raw `.md`) output and plugs into existing `_eval_` infrastructure (`prepare.py`, `agent_eval`, optional judge).

---

## End-to-end workflow

```text
[Long .md / PDF via Easy Dataset GUI]
        │
        ├─ question prompt (Type A/B/C/D)
        ├─ answer prompt   (reference + key_points)
        │
        v
[Easy Dataset project DB or manual export]
        │
        v
python -m _eval_.qa_eval.convert_easy_dataset \
    --project-dir path/to/easy-dataset/local-db/projects/<id> \
    --dataset-id codex-agentic \
    --out-root _eval_/datasets
        │
        v
_eval_/datasets/codex-agentic/
    corpus.jsonl
    queries.jsonl
    answers.jsonl          ← NEW (reference answers)
    qrels/test.tsv
    dataset.json           ← manifest (adapter hint)
        │
        v
python -m _eval_.qa_eval.run \
    --dataset codex-agentic \
    --profiles semantic baseline baseline_s2b full \
    --judge reference
        │
        v
results/qa_codex-agentic_<timestamp>.json
    mean: judge_correct | judge_complete | judge_grounded
    by_question_type: cross_span | deictic | paraphrase | fine_grained
```

---

## On-disk dataset layout

Register under `_eval_/datasets/<dataset_id>/`:

| File | Schema | Purpose |
|------|--------|---------|
| `corpus.jsonl` | `{"_id", "title", "text"}` | One row = one source document (full `.md` body) |
| `queries.jsonl` | `{"_id", "text", "question_type"?}` | Eval questions |
| `answers.jsonl` | `{"_id", "answer", "key_points"?, "question_type"?, "source_doc_id"?}` | Reference answers for judge |
| `qrels/test.tsv` | `query_id\tcorpus_id\tscore` | Doc-level gold (score `1` = relevant) |
| `dataset.json` | `{"dataset_id", "adapter", "corpus", "queries", "qrels", "answers"}` | Loader manifest |

**ID conventions**

- `doc_id`: slug from filename, e.g. `codex-agentic-world`
- `query_id`: `{doc_id}::q{index:04d}`, e.g. `codex-agentic-world::q0007`
- `answers.jsonl._id` == `queries.jsonl._id`

**Question types** (for stratified reporting)

| `question_type` | Targets profile knob |
|-----------------|----------------------|
| `cross_span` | `use_small_to_big` |
| `deictic` | `use_contextual` |
| `paraphrase` | `use_hyde`, `use_predict_questions` |
| `fine_grained` | `use_reranker`, semantic chunker |

---

## Module inventory

Create these files under `_eval_/qa_eval/`:

```text
_eval_/qa_eval/
├── __init__.py
├── types.py                 # TypedDicts shared across loader / judge / run
├── loader.py                # load corpus + queries + qrels + answers
├── convert_easy_dataset.py  # Easy Dataset SQLite → BEIR-ish files
├── build_from_md.py         # Optional: raw .md folder → intermediate JSONL
├── judge_reference.py       # Reference-answer judge (correct / complete / grounded)
├── prepare_qa.py              # PreparedEvalData + reference answers
└── run.py                     # CLI: index profiles, run agent/RAG, score
```

**Config hook** — add to `_eval_/config.py`:

```python
DATASETS["codex-agentic"] = DatasetSpec(
    dataset_id="codex-agentic",
    corpus="codex-agentic/corpus.jsonl",
    queries="codex-agentic/queries.jsonl",
    qrels="codex-agentic/qrels/test.tsv",
)
# answers path resolved via dataset.json manifest in loader.py
```

---

## 1. `types.py`

```python
"""Shared types for QA eval datasets."""

from __future__ import annotations

from typing import Literal, TypedDict

QuestionType = Literal["cross_span", "deictic", "paraphrase", "fine_grained", "unknown"]


class ReferenceAnswer(TypedDict, total=False):
    query_id: str
    answer: str
    key_points: list[str]
    question_type: QuestionType
    source_doc_id: str
    requires_cross_span: bool


class QaQueryMeta(TypedDict, total=False):
    question_type: QuestionType


class ReferenceJudgeVerdict(TypedDict):
    correct: bool
    complete: bool
    grounded: bool
    reason: str
```

---

## 2. `loader.py`

```python
"""Load QA eval files (BEIR core + reference answers)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from _eval_.data_preparing.beir import (
    CorpusDoc,
    EvalQuery,
    Qrels,
    QueryId,
    load_qrels,
    load_queries,
    iter_corpus,
)
from _eval_.paths import dataset_dir
from _eval_.qa_eval.types import QuestionType, ReferenceAnswer


@dataclass(frozen=True)
class QaDatasetManifest:
    dataset_id: str
    adapter: str
    corpus: str
    queries: str
    qrels: str
    answers: str


@dataclass(frozen=True)
class QaEvalData:
    pool: list[CorpusDoc]
    queries: dict[QueryId, EvalQuery]
    qrels: Qrels
    query_ids: list[QueryId]
    answers: dict[QueryId, ReferenceAnswer]
    question_types: dict[QueryId, QuestionType]


def load_manifest(dataset_id: str) -> QaDatasetManifest:
    root = dataset_dir(dataset_id)
    raw = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    return QaDatasetManifest(
        dataset_id=dataset_id,
        adapter=str(raw.get("adapter", "qa")),
        corpus=str(raw["corpus"]),
        queries=str(raw["queries"]),
        qrels=str(raw["qrels"]),
        answers=str(raw.get("answers", "answers.jsonl")),
    )


def load_answers(path: Path) -> dict[QueryId, ReferenceAnswer]:
    answers: dict[QueryId, ReferenceAnswer] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row["_id"]).strip()
            answers[qid] = ReferenceAnswer(
                query_id=qid,
                answer=str(row.get("answer", "")),
                key_points=[str(p) for p in row.get("key_points", [])],
                question_type=row.get("question_type", "unknown"),
                source_doc_id=str(row.get("source_doc_id", "")),
                requires_cross_span=bool(row.get("requires_cross_span", False)),
            )
    return answers


def load_qa_dataset(dataset_id: str) -> tuple[QaDatasetManifest, dict[QueryId, ReferenceAnswer]]:
    manifest = load_manifest(dataset_id)
    root = dataset_dir(dataset_id)
    answers = load_answers(root / manifest.answers)
    return manifest, answers
```

---

## 3. `convert_easy_dataset.py`

Reads Easy Dataset project storage and writes BEIR-ish files.

> **TODO**: Inspect actual Prisma/SQLite table names for your Easy Dataset version (`1.7.x`). Adjust `_TABLE_*` constants after `\`.tables\`` in the project DB.

```python
"""Convert an Easy Dataset project directory to QA eval files."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from _eval_.paths import dataset_dir

# --- adjust after inspecting local Easy Dataset schema ---
_TABLE_DOCUMENTS = "Document"       # TODO
_TABLE_CHUNKS = "Chunk"             # TODO (or TextSegment / SplitChunk)
_TABLE_QUESTIONS = "Question"       # TODO
_TABLE_DATASETS = "Dataset"         # TODO (QA pairs with answers)


@dataclass(frozen=True)
class EasyQuestionRow:
    query_id: str
    question: str
    answer: str
    doc_id: str
    question_type: str
    key_points: list[str]


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower())
    return re.sub(r"-+", "-", slug).strip("-") or "doc"


def iter_easy_questions(db_path: Path) -> list[EasyQuestionRow]:
    """Pull question/answer/doc linkage from Easy Dataset SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # TODO: replace with real JOIN once schema is confirmed
        sql = f"""
        SELECT
            q.id AS query_id,
            q.content AS question,
            d.answer AS answer,
            doc.fileName AS doc_name,
            q.questionType AS question_type
        FROM {_TABLE_QUESTIONS} q
        JOIN {_TABLE_DATASETS} d ON d.questionId = q.id
        JOIN {_TABLE_CHUNKS} c ON c.id = q.chunkId
        JOIN {_TABLE_DOCUMENTS} doc ON doc.id = c.documentId
        WHERE d.answer IS NOT NULL AND TRIM(d.answer) != ''
        """
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    out: list[EasyQuestionRow] = []
    for i, row in enumerate(rows):
        doc_id = slugify(str(row["doc_name"]))
        out.append(
            EasyQuestionRow(
                query_id=f"{doc_id}::q{i:04d}",
                question=str(row["question"]).strip(),
                answer=str(row["answer"]).strip(),
                doc_id=doc_id,
                question_type=str(row.get("question_type") or "unknown"),
                key_points=[],  # TODO: parse if answer prompt outputs JSON
            )
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_qrels(path: Path, pairs: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore"]
    lines.extend(f"{qid}\t{doc_id}\t{score}" for qid, doc_id, score in pairs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_project(
    *,
    project_db: Path,
    corpus_sources: dict[str, Path],
    dataset_id: str,
    out_root: Path | None = None,
) -> Path:
    """Write dataset files and return output directory."""
    questions = iter_easy_questions(project_db)
    out_dir = (out_root or dataset_dir(dataset_id)) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # corpus: one doc per source file
    corpus_rows: list[dict[str, object]] = []
    for doc_id, md_path in corpus_sources.items():
        text = md_path.read_text(encoding="utf-8")
        title = md_path.stem
        corpus_rows.append({"_id": doc_id, "title": title, "text": text})
    write_jsonl(out_dir / "corpus.jsonl", corpus_rows)

    query_rows: list[dict[str, object]] = []
    answer_rows: list[dict[str, object]] = []
    qrel_pairs: list[tuple[str, str, int]] = []

    for row in questions:
        query_rows.append(
            {"_id": row.query_id, "text": row.question, "question_type": row.question_type}
        )
        answer_rows.append(
            {
                "_id": row.query_id,
                "answer": row.answer,
                "key_points": row.key_points,
                "question_type": row.question_type,
                "source_doc_id": row.doc_id,
            }
        )
        qrel_pairs.append((row.query_id, row.doc_id, 1))

    write_jsonl(out_dir / "queries.jsonl", query_rows)
    write_jsonl(out_dir / "answers.jsonl", answer_rows)
    write_qrels(out_dir / "qrels" / "test.tsv", qrel_pairs)

    manifest = {
        "dataset_id": dataset_id,
        "adapter": "qa",
        "corpus": f"{dataset_id}/corpus.jsonl",
        "queries": f"{dataset_id}/queries.jsonl",
        "qrels": f"{dataset_id}/qrels/test.tsv",
        "answers": f"{dataset_id}/answers.jsonl",
    }
    (out_dir.parent / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Easy Dataset → QA eval files")
    parser.add_argument("--project-db", required=True, help="Path to project SQLite file")
    parser.add_argument("--corpus-md", action="append", default=[], help="doc_id=path/to/file.md")
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()

    corpus_sources: dict[str, Path] = {}
    for item in args.corpus_md:
        doc_id, _, path = item.partition("=")
        corpus_sources[doc_id.strip()] = Path(path.strip())

    out = convert_project(
        project_db=Path(args.project_db),
        corpus_sources=corpus_sources,
        dataset_id=args.dataset_id,
    )
    print(f"wrote dataset -> {out}")


if __name__ == "__main__":
    main()
```

**Inspect schema first**

```bash
sqlite3 path/to/project.db ".tables"
sqlite3 path/to/project.db ".schema Question"
```

---

## 4. `build_from_md.py` (optional fallback)

Use when you skip Easy Dataset export and keep intermediate JSONL in repo.

```python
"""Build intermediate QA JSONL from a folder of markdown files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _eval_.qa_eval.convert_easy_dataset import slugify, write_jsonl


def load_manual_pairs(path: Path) -> list[dict[str, object]]:
    """Each line: {"doc_id", "question", "answer", "question_type", "key_points"?}."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="manual_pairs.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    corpus_rows = []
    for md in sorted(args.md_dir.glob("*.md")):
        doc_id = slugify(md.stem)
        corpus_rows.append(
            {"_id": doc_id, "title": md.stem, "text": md.read_text(encoding="utf-8")}
        )
    write_jsonl(args.out / "corpus.jsonl", corpus_rows)

    pairs = load_manual_pairs(args.pairs)
    # TODO: fan out to queries.jsonl / answers.jsonl / qrels/test.tsv
    print(f"corpus docs={len(corpus_rows)} pairs={len(pairs)}")


if __name__ == "__main__":
    main()
```

---

## 5. `judge_reference.py`

Reference-answer mode for `_eval_.agent_eval.judge` — adds **`complete`**.

```python
"""LLM judge against reference answers (QA datasets)."""

from __future__ import annotations

import json
from typing import TypedDict

from llm.client import LLMClient

from _eval_.qa_eval.types import ReferenceAnswer, ReferenceJudgeVerdict

_JUDGE_REFERENCE_PROMPT = """You grade an assistant answer against a reference answer.

Question:
{query}

Reference answer:
{reference}

Key points that must appear for a complete answer:
{key_points}

Assistant answer:
{answer}

Judge:
- "correct": core facts match the reference; question is answered.
- "complete": all key points are covered (minor wording differences OK).
- "grounded": no claims that contradict the reference.

Return JSON only:
{{"correct": true|false, "complete": true|false, "grounded": true|false, "reason": "<one sentence>"}}
"""


async def judge_against_reference(
    llm: LLMClient,
    *,
    query: str,
    answer: str,
    reference: ReferenceAnswer,
) -> ReferenceJudgeVerdict:
    if not answer.strip():
        return ReferenceJudgeVerdict(
            correct=False, complete=False, grounded=False, reason="empty answer"
        )

    key_points = reference.get("key_points") or []
    prompt = _JUDGE_REFERENCE_PROMPT.format(
        query=query,
        reference=reference.get("answer", ""),
        key_points="\n".join(f"- {p}" for p in key_points) or "(none listed)",
        answer=answer,
    )
    try:
        response = await llm.arequest_llm(
            [{"role": "user", "content": prompt}],
            json_output=True,
        )
        data = json.loads(response.content or "{}")
        return ReferenceJudgeVerdict(
            correct=bool(data.get("correct", False)),
            complete=bool(data.get("complete", False)),
            grounded=bool(data.get("grounded", False)),
            reason=str(data.get("reason", "")),
        )
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        return ReferenceJudgeVerdict(
            correct=False, complete=False, grounded=False, reason=f"judge error: {exc}"
        )
```

**Wire into** `_eval_/agent_eval/pipeline.py`:

```python
# if cfg.use_reference_judge:
#     verdict = await judge_against_reference(llm, query=..., answer=..., reference=answers[qid])
#     metrics["judge_correct"] = float(verdict["correct"])
#     metrics["judge_complete"] = float(verdict["complete"])
#     metrics["judge_grounded"] = float(verdict["grounded"])
```

---

## 6. `prepare_qa.py`

Wraps existing `prepare_eval_data`; adds reference answers.

```python
"""Prepare QA eval data: BEIR pool + reference answers."""

from __future__ import annotations

from dataclasses import dataclass

from _eval_.config import RunConfig
from _eval_.data_preparing.prepare import prepare_eval_data
from _eval_.data_preparing.beir import QueryId
from _eval_.qa_eval.loader import load_qa_dataset
from _eval_.qa_eval.types import ReferenceAnswer, QuestionType


@dataclass(frozen=True)
class PreparedQaEvalData:
    pool: list
    queries: dict
    qrels: dict
    query_ids: list[QueryId]
    answers: dict[QueryId, ReferenceAnswer]
    question_types: dict[QueryId, QuestionType]


def prepare_qa_eval_data(cfg: RunConfig) -> PreparedQaEvalData:
    base = prepare_eval_data(cfg)
    _, answers = load_qa_dataset(cfg.dataset)
    question_types = {
        qid: answers[qid].get("question_type", "unknown")
        for qid in base.query_ids
        if qid in answers
    }
    return PreparedQaEvalData(
        pool=base.pool,
        queries=base.queries,
        qrels=base.qrels,
        query_ids=base.query_ids,
        answers=answers,
        question_types=question_types,
    )
```

---

## 7. `run.py`

Profile ablation CLI. Reuses `rag_eval` indexing + `agent_eval` scoring patterns.

```python
"""QA eval CLI: compare RAG profiles on answer quality."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

from _eval_.config import RunConfig, collection_name, load_rag_eval_config
from _eval_.paths import REPO_ROOT, results_dir
from _eval_.qa_eval.prepare_qa import prepare_qa_eval_data
from _eval_.scoring.metrics import mean_metrics

# TODO: import build_RAG_indexer/retriever, evaluate_pattern or a slim answer-only runner


def _bucket_means(per_query: list[dict], question_types: dict[str, str]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in per_query:
        qtype = question_types.get(row["query_id"], "unknown")
        buckets[qtype].append(row["metrics"])
    return {qtype: mean_metrics(rows) for qtype, rows in buckets.items()}


async def run_qa_eval(cfg: RunConfig, *, judge_mode: str) -> dict[str, object]:
    prepared = prepare_qa_eval_data(cfg)
    print(
        f"dataset={cfg.dataset} queries={len(prepared.query_ids)} "
        f"pool_docs={len(prepared.pool)} judge={judge_mode}"
    )

    all_results: dict[str, object] = {"profiles": {}}

    for profile_id in cfg.profiles:
        collection = collection_name(cfg.dataset, profile_id)
        # TODO:
        # 1. index prepared.pool with build_RAG_indexer(profile flags)
        # 2. for each query: retriever.aquery → LLM answer (or agent_eval react)
        # 3. judge_against_reference(...)
        # 4. collect per_query metrics + ctx_* from RecordingRetriever
        all_results["profiles"][profile_id] = {
            "mean_metrics": {},
            "by_question_type": {},
            "per_query": [],
        }

    return all_results


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="QA eval for RAG profile ablation")
    parser.add_argument("--config", default=None, help="rag_eval_arg_config.yaml override")
    parser.add_argument(
        "--judge",
        choices=["reference", "passage", "none"],
        default="reference",
        help="reference=QA gold answer; passage=legacy BEIR judge",
    )
    args = parser.parse_args()

    cfg = load_rag_eval_config(args.config) if args.config else load_rag_eval_config()
    payload = asyncio.run(run_qa_eval(cfg, judge_mode=args.judge))

    out = results_dir() / f"qa_{cfg.dataset}_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
```

---

## Example `dataset.json`

Place at `_eval_/datasets/codex-agentic/dataset.json`:

```json
{
  "dataset_id": "codex-agentic",
  "adapter": "qa",
  "corpus": "codex-agentic/corpus.jsonl",
  "queries": "codex-agentic/queries.jsonl",
  "qrels": "codex-agentic/qrels/test.tsv",
  "answers": "codex-agentic/answers.jsonl"
}
```

## Example `answers.jsonl` row

```json
{
  "_id": "codex-agentic-world::q0003",
  "answer": "该团队刻意限制所有代码由 Codex 生成，以验证当工程工作从写代码转向设计环境、明确意图和构建反馈回路时，交付速度能提升一个数量级。",
  "key_points": [
    "所有代码由 Codex 生成",
    "工程角色转向环境设计与反馈回路",
    "目标是数量级提升交付速度"
  ],
  "question_type": "deictic",
  "source_doc_id": "codex-agentic-world",
  "requires_cross_span": false
}
```

---

## Success criteria (what "winning" looks like)

Report **by question type**, not one global average:

| Subset | Expect `baseline_s2b` > `baseline` on | Metric |
|--------|----------------------------------------|--------|
| `cross_span` | answer completeness | `judge_complete` |
| `deictic` | factual correctness | `judge_correct` |
| `paraphrase` | retrieval + answer | `judge_correct`, `ctx_recall@10` |
| `fine_grained` | precision | `judge_correct`, `ctx_ndcg@3` |

Example result snippet:

```json
{
  "profiles": {
    "baseline": {
      "by_question_type": {
        "cross_span": {"judge_complete": 0.52},
        "deictic": {"judge_correct": 0.61}
      }
    },
    "baseline_s2b": {
      "by_question_type": {
        "cross_span": {"judge_complete": 0.78},
        "deictic": {"judge_correct": 0.63}
      }
    }
  }
}
```

---

## Implementation checklist

- [ ] Inspect Easy Dataset SQLite schema; fix `_TABLE_*` and SQL in `convert_easy_dataset.py`
- [ ] Add `codex-agentic` (or your id) to `_eval_/config.py` `DATASETS`
- [ ] Implement `judge_reference.py` + wire `--judge reference` in agent pipeline
- [ ] Flesh out `run.py` indexing loop (copy from `_eval_/rag_eval/pipeline.py`)
- [ ] Add 10-query smoke set under `_eval_/datasets/smoke/` with one `.md` + manual pairs
- [ ] Notebook or script to render profile × question_type heatmap

---

## Quick smoke (once wired)

```bash
# 1) Convert Easy Dataset project
python -m _eval_.qa_eval.convert_easy_dataset \
  --project-db "C:/path/to/easy-dataset/local-db/projects/<id>/project.db" \
  --dataset-id codex-agentic \
  --corpus-md "codex-agentic-world=get_start/工程技术：在智能体优先的世界中利用 Codex.md"

# 2) Run QA eval
python -m _eval_.qa_eval.run \
  --config rag_eval_arg_config.yaml \
  --judge reference
```

Set `rag_eval_arg_config.yaml`:

```yaml
dataset: codex-agentic
profiles: [semantic, baseline, baseline_s2b, baseline_predict_q, full]
query_limit: 20
pool:
  rel_threshold: 1
  max_distractors_per_query: 5
```
