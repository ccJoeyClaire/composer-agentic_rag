"""Dataset registry and typed run configuration for RAG eval.

RAG eval harness parameters load from ``rag_eval_arg_config.yaml`` at the repo
root. Profile *definitions* (chunker/reranker flags) stay in ``arg_config.yaml``
and are read directly by :mod:`_eval_.rag_eval.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from _eval_.data_preparing.pooling import PoolSpec
from _eval_.paths import REPO_ROOT, dataset_dir

_DEFAULT_RAG_EVAL_CONFIG = REPO_ROOT / "rag_eval_arg_config.yaml"

# Default metric cutoffs and retrieval depth. Kept here so run + pipeline agree.
DEFAULT_K_VALUES: tuple[int, ...] = (3, 10, 20)
DEFAULT_CHUNK_FETCH_MULTIPLIER = 4
DEFAULT_QUERY_LIMIT = 5
DEFAULT_INDEX_CONCURRENCY = 8
DEFAULT_PREDICT_QUESTION_MAX_CONCURRENCY = 15


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's three BEIR files live (relative to its dataset dir)."""

    dataset_id: str
    corpus: str
    queries: str
    qrels: str

    def corpus_path(self) -> Path:
        return dataset_dir(self.dataset_id) / self.corpus

    def queries_path(self) -> Path:
        return dataset_dir(self.dataset_id) / self.queries

    def qrels_path(self) -> Path:
        return dataset_dir(self.dataset_id) / self.qrels


DATASETS: dict[str, DatasetSpec] = {
    "trec-covid": DatasetSpec(
        dataset_id="trec-covid",
        corpus="trec-covid/corpus.jsonl",
        queries="trec-covid/queries.jsonl",
        qrels="trec-covid/qrels/test.tsv",
    ),
    "msmarco": DatasetSpec(
        dataset_id="msmarco",
        corpus="msmarco/corpus.jsonl",
        queries="msmarco/queries.jsonl",
        qrels="msmarco/qrels/dev.tsv",
    ),
}


@dataclass(frozen=True)
class RunConfig:
    """Everything one RAG eval run needs; built from ``rag_eval_arg_config.yaml``."""

    dataset: str
    profiles: list[str]
    pool_spec: PoolSpec
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    query_limit: int | None = DEFAULT_QUERY_LIMIT
    chunk_fetch_multiplier: int = DEFAULT_CHUNK_FETCH_MULTIPLIER
    index_concurrency: int = DEFAULT_INDEX_CONCURRENCY
    predict_question_max_concurrency: int = DEFAULT_PREDICT_QUESTION_MAX_CONCURRENCY
    recreate: bool = True

    @property
    def max_k(self) -> int:
        return max(self.k_values)

    @property
    def fetch_chunks(self) -> int:
        """How many chunks to pull from the retriever before dedup-to-doc."""
        return self.max_k * self.chunk_fetch_multiplier


# Default per-call retrieval depth for the agent's RAG tool.
DEFAULT_AGENT_RAG_TOP_K = 10
DEFAULT_RECURSION_LIMIT = 50

AGENT_SYSTEM_PROMPT = (
    "You are a biomedical question-answering assistant. Before answering, call "
    "RAG_search_tool to retrieve evidence from the knowledge base. Base your "
    "answer ONLY on the retrieved passages; if the evidence is insufficient, say "
    "so explicitly. End with a concise, direct final answer."
)


@dataclass(frozen=True)
class AgentRunConfig:
    """Config for an agent eval run comparing reflection patterns."""

    dataset: str
    patterns: list[str]
    rag_profile: str
    pool_spec: PoolSpec
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    query_limit: int | None = DEFAULT_QUERY_LIMIT
    agent_rag_top_k: int = DEFAULT_AGENT_RAG_TOP_K
    index_concurrency: int = DEFAULT_INDEX_CONCURRENCY
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    recreate: bool = True
    use_judge: bool = False

    @property
    def max_k(self) -> int:
        return max(self.k_values)


def collection_name(dataset: str, profile_id: str) -> str:
    """Qdrant collection name; namespaced to avoid clobbering legacy eval runs."""
    return f"pooleval_{dataset}_{profile_id}"


def _require_mapping(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError(
            f"rag_eval_arg_config.yaml: root must be a mapping, got {type(data).__name__}"
        )
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"rag_eval_arg_config.yaml: missing or invalid section {name!r}")
    return section


def load_rag_eval_config(path: Path | None = None) -> RunConfig:
    """Load RAG eval harness settings from ``rag_eval_arg_config.yaml``.

    Args:
        path: Override config file path (defaults to repo-root file).

    Returns:
        Typed :class:`RunConfig` for one eval run.
    """
    config_path = path or _DEFAULT_RAG_EVAL_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"rag_eval_arg_config.yaml: root must be a mapping, got {type(data).__name__}"
        )

    pool_raw = _require_mapping(data, "pool")
    profiles_raw = data.get("profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ValueError("rag_eval_arg_config.yaml: profiles must be a non-empty list")

    k_values_raw = data.get("k_values", list(DEFAULT_K_VALUES))
    if not isinstance(k_values_raw, list) or not k_values_raw:
        raise ValueError("rag_eval_arg_config.yaml: k_values must be a non-empty list")

    query_limit = data.get("query_limit", DEFAULT_QUERY_LIMIT)
    if query_limit is not None:
        query_limit = int(query_limit)

    max_distractors = pool_raw.get("max_distractors_per_query")
    if max_distractors is not None:
        max_distractors = int(max_distractors)

    return RunConfig(
        dataset=str(data["dataset"]),
        profiles=[str(p) for p in profiles_raw],
        pool_spec=PoolSpec(
            rel_threshold=int(pool_raw.get("rel_threshold", 1)),
            max_distractors_per_query=max_distractors,
        ),
        k_values=tuple(sorted({int(k) for k in k_values_raw})),
        query_limit=query_limit,
        chunk_fetch_multiplier=int(
            data.get("chunk_fetch_multiplier", DEFAULT_CHUNK_FETCH_MULTIPLIER)
        ),
        index_concurrency=int(
            data.get("index_concurrency", DEFAULT_INDEX_CONCURRENCY)
        ),
        predict_question_max_concurrency=int(
            data.get(
                "predict_question_max_concurrency",
                DEFAULT_PREDICT_QUESTION_MAX_CONCURRENCY,
            )
        ),
        recreate=bool(data.get("recreate", True)),
    )
