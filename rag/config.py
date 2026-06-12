"""Load typed pipeline settings from ``config.yaml`` at the repo root."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "config.yaml"

_PROFILE_BOOL_FIELDS = (
    "use_contextual",
    "use_small_to_big",
    "use_predict_questions",
    "use_hyde",
    "use_reranker",
)


@dataclass(frozen=True)
class ChunkerConfig:
    """Typed view of ``config.yaml`` → ``chunker``; values come only from YAML."""

    chunk_tokens: int
    overlap_tokens: int
    break_similarity: float
    min_chunk_tokens: int


@dataclass(frozen=True)
class RetrieverConfig:
    """Typed view of ``config.yaml`` → ``retriever``; values come only from YAML."""

    recall_n: int
    top_k: int


@dataclass(frozen=True)
class ProfileConfig:
    """Typed view of one ``profiles.<id>`` entry."""

    use_contextual: bool
    use_small_to_big: bool
    use_predict_questions: bool
    use_hyde: bool
    use_reranker: bool


@dataclass(frozen=True)
class RagConfig:
    chunker: ChunkerConfig
    retriever: RetrieverConfig
    profiles: Dict[str, ProfileConfig]


def _config_section(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError(
            f"config.yaml: root must be a mapping, got {type(data).__name__}"
        )
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"config.yaml: missing or invalid section {name!r}")
    return section


def _config_value(section: dict[str, object], section_name: str, key: str) -> object:
    if key not in section:
        raise KeyError(f"config.yaml: {section_name}.{key} is required")
    return section[key]


def _parse_profile(raw: object) -> ProfileConfig:
    data = raw if isinstance(raw, dict) else {}
    # Profile flags are optional in YAML; omitted key means false (not a second default table).
    return ProfileConfig(
        **{key: bool(data.get(key, False)) for key in _PROFILE_BOOL_FIELDS}
    )


@lru_cache(maxsize=1)
def get_rag_config() -> RagConfig:
    """Load pipeline settings from ``config.yaml`` (single source of truth)."""
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    chunker_raw = _config_section(data, "chunker")
    retriever_raw = _config_section(data, "retriever")
    profiles_raw = _config_section(data, "profiles")

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
    profiles = {
        profile_id: _parse_profile(cfg)
        for profile_id, cfg in profiles_raw.items()
    }
    return RagConfig(chunker=chunker, retriever=retriever, profiles=profiles)
