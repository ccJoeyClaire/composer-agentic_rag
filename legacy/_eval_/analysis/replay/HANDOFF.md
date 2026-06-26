# B-line replay tools — implementation handoff

Post-eval **reproduction** of index state and query pipelines. **Do not hook into `rag_eval/run.py` or eval pipeline.**

Context: NFCorpus profile ablation (`arg_config.yaml` profiles like `baseline`, `baseline_s2b`, …).
Collections already exist after eval: `pooleval_{dataset}_{profile_id}` (see `_eval_.config.collection_name`).

---

## Scope (3 deliverables)

### 1. `QdrantVectorStore.alist_chunks` + `inspect_collection` CLI

**File:** `rag/store/qdrant_store.py` — add method:

```python
async def alist_chunks(
    self,
    *,
    limit: int = 20,
    offset: str | None = None,
    doc_id: str | None = None,
    source: str | None = None,
    with_vectors: bool = False,
) -> tuple[list[Chunk], str | None]:
    """Scroll collection payload; optional filter on metadata.doc_id or metadata.source."""
```

- Reuse `scroll` pattern from `acount_by_source` (filter on `_PAYLOAD_METADATA.doc_id` / `.source`).
- Return `(chunks, next_offset)` for pagination.
- Add `BaseVectorStore` abstract stub or `NotImplementedError` default if needed for fakes.

**File:** `_eval_/analysis/replay/inspect_collection.py`

```bash
python -m _eval_.analysis.replay.inspect_collection \
  --collection pooleval_nfcorpus_semantic \
  --limit 20 \
  [--doc-id DOC_ID] [--source TITLE] [--json] [--summary-only]
```

**`summarize_chunks(chunks) -> IndexSummary`** (TypedDict or dataclass):

| Field | Purpose |
|-------|---------|
| `total_listed` | len(chunks) in this page |
| `chunks_per_doc` | Counter by `metadata.doc_id` |
| `heading_path_nonempty_ratio` | contextual / section signal |
| `chunk_role_counts` | s2b: small vs parent |
| `avg_content_len` / `avg_embed_text_len` | contextual / predict_q inflation |

Print human table + optional JSON. `--summary-only` scrolls more points (e.g. cap 2000) for stats only.

---

### 2. `query` CLI (replay query pipeline)

**File:** `_eval_/analysis/replay/query.py`

```bash
python -m _eval_.analysis.replay.query \
  --profile baseline \
  [--collection pooleval_nfcorpus_baseline] \
  [--dataset nfcorpus] \
  --query "..." \
  [--query-id PLAIN-2] \
  [--top-k 80] \
  [--stages retrieved,reranked,final] \
  [--gold-doc-ids id1,id2] \
  [--json]
```

**Build retriever:** reuse `beir_runner.load_profile_flags` + `build_RAG_retriever(collection, flags…, store=existing)`.
- Open `QdrantVectorStore(collection=…)` **without re-indexing**.
- Call `retriever.aquery_trace(query, top_k=top_k)` (already in `rag/core.py`).

**Output per stage** (`TRACE_RETRIEVED_KEY`, `TRACE_RERANKED_KEY`, final `result.chunks`):

```
rank | doc_id | chunk_role | score | content_preview(80)
```

If `--gold-doc-ids` or `--query-id` (+ load qrels from `DATASETS[dataset]`):

```
gold doc_id | rank@retrieved | rank@reranked | rank@final | in_top_k
```

**Optional:** `--hyde-from-log PATH` — for profiles with hyde, embed recorded `hyde_document` instead of calling LLM (replay exact eval-time retrieval). Phase 2; stub OK.

Shared helper (avoid duplicating `beir_runner`):

```python
# _eval_/analysis/replay/rag_factory.py
def open_retriever(profile_id: str, collection: str) -> RAGRetriever: ...
def resolve_collection(dataset: str, profile_id: str, collection: str | None) -> str: ...
```

---

### 3. HyDE call log (append JSONL at transform time)

**File:** `rag/query_transformer/hyde.py`

After successful `atransform`, append one JSON line:

```python
class HydeLogEntry(TypedDict):
    timestamp_utc: str
    query: str
    hyde_document: str
    profile: str | None      # optional context passed by caller
    collection: str | None
```

- Path: `os.environ.get("HYDE_LOG_PATH")` or default `_eval_/hyde_log/hyde.jsonl` (mkdir parents).
- Env `HYDE_LOG=0` disables.
- **Do not break** when path unset in tests — default off in unit tests or use tmp path in conftest.

`replay.query` with `--profile baseline_hyde` uses live LLM by default; document that JSONL enables exact replay later.

---

## Explicit non-goals

- No eval pipeline hooks / no sidecar snapshot dirs.
- No changes to `rag_eval/run.py` (smoke config is optional separate yaml).
- No ipynb work (A-line done in `_eval_/analysis/visual/` + `nfcorpus.ipynb`).

---

## Tests

| Test | Location |
|------|----------|
| `alist_chunks` filter + limit | `tests/rag/test_qdrant_store.py` (extend) or fake scroll |
| `summarize_chunks` | `tests/eval/test_replay_inspect.py` |
| query rank extraction from trace lists | `tests/eval/test_replay_query.py` (mock retriever / fake chunks) |
| HyDE append line | `tests/rag/test_hyde.py` — tmp file, one line JSON |

Use existing `tests/fakes/vector_store.py` patterns; full Qdrant integration optional behind marker.

---

## Smoke workflow (document in module docstrings)

1. Copy `rag_eval_arg_config.yaml` → `rag_eval_smoke_arg_config.yaml`: `query_limit: 5`, 4 profiles `[token, semantic, baseline, baseline_s2b]`, `max_distractors_per_query: 20`.
2. `python -m _eval_.rag_eval.run` (with smoke config if `--config` added later, or manual swap).
3. `inspect_collection` on `token` vs `semantic` — chunks/doc should differ if paragraph structure exists.
4. `replay.query` on 1–2 queries with `semantic_rerank` — compare retrieved vs reranked gold ranks.

---

## Key references

| Item | Path |
|------|------|
| Trace API | `RAGRetriever.aquery_trace` — `rag/core.py` |
| TRACE keys | `rag/base.py`: `TRACE_RETRIEVED_KEY`, `TRACE_RERANKED_KEY`, `TRACE_HYDE_DOCUMENT_KEY` |
| Profile flags | `arg_config.yaml` → `profiles.<id>` |
| Collection naming | `_eval_.config.collection_name` |
| Chunk metadata keys | `rag/document_augmentation/parent_builder.py` constants |
| Existing profile loader | `_eval_.reflection_eval.beir_runner.load_profile_flags` |

---

## NFCorpus eval note (why these tools matter)

Last run: `token ≈ semantic` → inspect chunks/doc first.
`semantic_rerank < semantic` → `replay.query` retrieved vs reranked.
HyDE unstable → JSONL required for exact replay.

Results JSON for A-line viz: `_eval_/results/nfcorpus_20260618T045954.json`.
