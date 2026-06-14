"""Dataset registry and the typed run configuration.

Adding a new BEIR dataset = one entry in :data:`DATASETS`. Everything else
(loaders, pooling, metrics, runner) is dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from _eval_.paths import dataset_dir
from _eval_.pooling import PoolSpec

# Default metric cutoffs and retrieval depth. Kept here so run + pipeline agree.
DEFAULT_K_VALUES: tuple[int, ...] = (3, 10, 20)
# Fetch this many * max(k) chunks so dedup-to-doc still yields enough unique docs.
DEFAULT_CHUNK_FETCH_MULTIPLIER = 4
DEFAULT_QUERY_LIMIT = 5
DEFAULT_INDEX_CONCURRENCY = 8


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


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {dataset_id!r}. Supported: {supported}")
    return DATASETS[dataset_id]


@dataclass(frozen=True)
class RunConfig:
    """Everything one eval run needs; built once in :mod:`_eval_.run`."""

    dataset: str
    profiles: list[str]
    pool_spec: PoolSpec
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    query_limit: int | None = DEFAULT_QUERY_LIMIT
    chunk_fetch_multiplier: int = DEFAULT_CHUNK_FETCH_MULTIPLIER
    index_concurrency: int = DEFAULT_INDEX_CONCURRENCY
    in_memory: bool = True
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

# Steers the agent to ground answers in retrieved passages so context-recall and
# grounding signals are meaningful.
AGENT_SYSTEM_PROMPT = (
    "You are a biomedical question-answering assistant. Before answering, call "
    "RAG_search_tool to retrieve evidence from the knowledge base. Base your "
    "answer ONLY on the retrieved passages; if the evidence is insufficient, say "
    "so explicitly. End with a concise, direct final answer."
)


@dataclass(frozen=True)
class AgentRunConfig:
    """Config for an agent eval run comparing reflection patterns.

    All patterns share one pooled index served through a single ``rag_profile``
    retrieval backend, so retrieval quality is held fixed while the agent control
    flow (react / crag / self_rag / ...) varies.
    """

    dataset: str
    patterns: list[str]
    rag_profile: str
    pool_spec: PoolSpec
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    query_limit: int | None = DEFAULT_QUERY_LIMIT
    agent_rag_top_k: int = DEFAULT_AGENT_RAG_TOP_K
    index_concurrency: int = DEFAULT_INDEX_CONCURRENCY
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    in_memory: bool = True
    recreate: bool = True
    use_judge: bool = False

    @property
    def max_k(self) -> int:
        return max(self.k_values)


def collection_name(dataset: str, profile_id: str) -> str:
    """Qdrant collection name; namespaced to avoid clobbering legacy eval runs."""
    return f"pooleval_{dataset}_{profile_id}"
