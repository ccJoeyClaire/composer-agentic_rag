"""Load typed pipeline settings from ``arg_config.yaml``.

Run (from repo root):
  python -m rag.config
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "arg_config.yaml"

_PROFILE_BOOL_FIELDS = (
    "use_token_chunker",
    "use_contextual",
    "use_small_to_big",
    "use_predict_questions",
    "use_hyde",
    "use_reranker",
)

DEFAULT_PROFILE_ID = "baseline"


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
class ProfileConfig:
    """Typed view of one ``profiles.<id>`` entry."""

    use_token_chunker: bool
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


def default_config_path() -> Path:
    """Return the repo-root ``arg_config.yaml`` path."""
    return _DEFAULT_CONFIG_PATH


def resolve_config_path(config_path: Path | None) -> Path:
    """Resolve ``None`` to :func:`default_config_path`."""
    return config_path if config_path is not None else default_config_path()


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


def _parse_profile(raw: object) -> ProfileConfig:
    data = raw if isinstance(raw, dict) else {}
    return ProfileConfig(
        **{key: bool(data.get(key, False)) for key in _PROFILE_BOOL_FIELDS}
    )


@lru_cache(maxsize=8)
def load_rag_config(config_path: Path) -> RagConfig:
    """Load pipeline settings from an explicit YAML path."""
    with config_path.open(encoding="utf-8") as handle:
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


def get_rag_config(config_path: Path | None = None) -> RagConfig:
    """Load pipeline settings; ``config_path`` defaults to repo-root YAML."""
    return load_rag_config(resolve_config_path(config_path))


def get_profile(
    config: RagConfig,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> ProfileConfig:
    """Return one named profile from a loaded :class:`RagConfig`."""
    try:
        return config.profiles[profile_id]
    except KeyError as exc:
        known = ", ".join(sorted(config.profiles))
        raise KeyError(f"Unknown profile {profile_id!r}; known: {known}") from exc


if __name__ == "__main__":
    def demo_print_config() -> None:
        """Offline smoke: print parsed chunker, retriever, and profile flags."""
        cfg = get_rag_config()
        path = default_config_path()
        print(f"Config path: {path}")
        print("\n=== chunker ===")
        print(f"  chunk_tokens={cfg.chunker.chunk_tokens}")
        print(f"  overlap_tokens={cfg.chunker.overlap_tokens}")
        print(f"  break_similarity={cfg.chunker.break_similarity}")
        print(f"  min_chunk_tokens={cfg.chunker.min_chunk_tokens}")
        print("\n=== retriever ===")
        print(f"  recall_n={cfg.retriever.recall_n}")
        print(f"  top_k={cfg.retriever.top_k}")
        print("\n=== profiles ===")
        for pid, profile in sorted(cfg.profiles.items()):
            flags = ", ".join(
                key for key in _PROFILE_BOOL_FIELDS if getattr(profile, key)
            )
            print(f"  {pid}: {flags or '(all false)'}")

    demo_print_config()
