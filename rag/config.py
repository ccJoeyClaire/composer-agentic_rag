"""Load typed pipeline settings from ``arg_config.yaml``.

Run (from repo root):
  python -m rag.config
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "arg_config.yaml"

_INDEX_BOOL_FIELDS = (
    "use_token_chunker",
    "use_contextual",
    "use_small_to_big",
    "use_predict_questions",
)

_RETRIEVE_BOOL_FIELDS = (
    "use_contextual",
    "use_small_to_big",
    "use_hyde",
    "use_reranker",
)

DEFAULT_INDEX_PROFILE_ID = "baseline"
DEFAULT_RETRIEVE_PROFILE_ID = "rerank_contextual"

# Back-compat alias for callers not yet migrated.
DEFAULT_PROFILE_ID = DEFAULT_INDEX_PROFILE_ID


@dataclass(frozen=True)
class ChunkerConfig:
    """Typed view of ``arg_config.yaml`` → ``chunker``; values come only from YAML."""

    chunk_tokens: int
    overlap_tokens: int
    break_similarity: float
    min_chunk_tokens: int


@dataclass(frozen=True)
class RetrieverConfig:
    """Typed view of ``arg_config.yaml`` → ``retriever``; values come only from YAML."""

    recall_n: int
    top_k: int


@dataclass(frozen=True)
class IndexProfileConfig:
    """Typed view of one ``index_profiles.<id>`` entry."""

    use_token_chunker: bool
    use_contextual: bool
    use_small_to_big: bool
    use_predict_questions: bool


@dataclass(frozen=True)
class RetrieveProfileConfig:
    """Typed view of one ``retrieve_profiles.<id>`` entry."""

    use_contextual: bool
    use_small_to_big: bool
    use_hyde: bool
    use_reranker: bool


@dataclass(frozen=True)
class RagConfig:
    chunker: ChunkerConfig
    retriever: RetrieverConfig
    index_profiles: Dict[str, IndexProfileConfig]
    retrieve_profiles: Dict[str, RetrieveProfileConfig]


def _config_section(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError(
            f"arg_config.yaml: root must be a mapping, got {type(data).__name__}"
        )
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"arg_config.yaml: missing or invalid section {name!r}")
    return section


def _config_value(section: dict[str, object], section_name: str, key: str) -> object:
    if key not in section:
        raise KeyError(f"arg_config.yaml: {section_name}.{key} is required")
    return section[key]


def _parse_index_profile(raw: object) -> IndexProfileConfig:
    data = raw if isinstance(raw, dict) else {}
    return IndexProfileConfig(
        **{key: bool(data.get(key, False)) for key in _INDEX_BOOL_FIELDS}
    )


def _parse_retrieve_profile(raw: object) -> RetrieveProfileConfig:
    data = raw if isinstance(raw, dict) else {}
    return RetrieveProfileConfig(
        **{key: bool(data.get(key, False)) for key in _RETRIEVE_BOOL_FIELDS}
    )


@lru_cache(maxsize=8)
def load_rag_config(config_path: Path) -> RagConfig:
    """Load pipeline settings from an explicit YAML path."""
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    chunker_raw = _config_section(data, "chunker")
    retriever_raw = _config_section(data, "retriever")
    index_profiles_raw = _config_section(data, "index_profiles")
    retrieve_profiles_raw = _config_section(data, "retrieve_profiles")

    chunker = ChunkerConfig(
        chunk_tokens=int(_config_value(chunker_raw, "chunker", "chunk_tokens")),
        overlap_tokens=int(_config_value(chunker_raw, "chunker", "overlap_tokens")),
        break_similarity=float(
            _config_value(chunker_raw, "chunker", "break_similarity")
        ),
        min_chunk_tokens=int(
            _config_value(chunker_raw, "chunker", "min_chunk_tokens")
        ),
    )
    retriever = RetrieverConfig(
        recall_n=int(_config_value(retriever_raw, "retriever", "recall_n")),
        top_k=int(_config_value(retriever_raw, "retriever", "top_k")),
    )
    index_profiles = {
        profile_id: _parse_index_profile(cfg)
        for profile_id, cfg in index_profiles_raw.items()
    }
    retrieve_profiles = {
        profile_id: _parse_retrieve_profile(cfg)
        for profile_id, cfg in retrieve_profiles_raw.items()
    }
    return RagConfig(
        chunker=chunker,
        retriever=retriever,
        index_profiles=index_profiles,
        retrieve_profiles=retrieve_profiles,
    )


def get_rag_config(config_path: Path | None = None) -> RagConfig:
    if config_path is not None:
        return load_rag_config(config_path)
    env_path = os.environ.get("RAG_CONFIG_PATH")
    path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH
    return load_rag_config(path)


def get_index_profile(
    config: RagConfig,
    profile_id: str = DEFAULT_INDEX_PROFILE_ID,
) -> IndexProfileConfig:
    """Return one named index profile from a loaded :class:`RagConfig`."""
    try:
        return config.index_profiles[profile_id]
    except KeyError as exc:
        known = ", ".join(sorted(config.index_profiles))
        raise KeyError(f"Unknown index profile {profile_id!r}; known: {known}") from exc


def get_retrieve_profile(
    config: RagConfig,
    profile_id: str = DEFAULT_RETRIEVE_PROFILE_ID,
) -> RetrieveProfileConfig:
    """Return one named retrieve profile from a loaded :class:`RagConfig`."""
    try:
        return config.retrieve_profiles[profile_id]
    except KeyError as exc:
        known = ", ".join(sorted(config.retrieve_profiles))
        raise KeyError(
            f"Unknown retrieve profile {profile_id!r}; known: {known}"
        ) from exc


if __name__ == "__main__":
    def demo_print_config() -> None:
        """Offline smoke: print parsed chunker, retriever, and profile flags."""
        cfg = get_rag_config()
        path = _DEFAULT_CONFIG_PATH
        print(f"Config path: {path}")
        print("\n=== chunker ===")
        print(f"  chunk_tokens={cfg.chunker.chunk_tokens}")
        print(f"  overlap_tokens={cfg.chunker.overlap_tokens}")
        print(f"  break_similarity={cfg.chunker.break_similarity}")
        print(f"  min_chunk_tokens={cfg.chunker.min_chunk_tokens}")
        print("\n=== retriever ===")
        print(f"  recall_n={cfg.retriever.recall_n}")
        print(f"  top_k={cfg.retriever.top_k}")
        print("\n=== index_profiles ===")
        for pid, profile in sorted(cfg.index_profiles.items()):
            flags = ", ".join(
                key for key in _INDEX_BOOL_FIELDS if getattr(profile, key)
            )
            print(f"  {pid}: {flags or '(all false)'}")
        print("\n=== retrieve_profiles ===")
        for pid, profile in sorted(cfg.retrieve_profiles.items()):
            flags = ", ".join(
                key for key in _RETRIEVE_BOOL_FIELDS if getattr(profile, key)
            )
            print(f"  {pid}: {flags or '(all false)'}")

    demo_print_config()
