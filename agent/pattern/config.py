"""Load reflection patterns from ``agent_arg_config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "agent_arg_config.yaml"

_PATTERN_BOOL_FIELDS = (
    "enable_retrieval_gate",
    "enable_rag_profile_router",
    "enable_human_feedback",
)


@dataclass(frozen=True)
class RagContextConfig:
    """Global RAG context budget for the LLM view (not per-pattern)."""

    max_chunks: int | None = None


@dataclass(frozen=True)
class PatternConfig:
    """Capability toggles for one named reflection pattern."""

    enable_retrieval_gate: bool
    enable_rag_profile_router: bool
    enable_human_feedback: bool
    system_prompt_key: str = "default"


@dataclass(frozen=True)
class AgentPatternConfig:
    """Root document loaded from ``agent_arg_config.yaml``."""

    patterns: Dict[str, PatternConfig]
    rag_context: RagContextConfig


def _config_section(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError(
            f"agent_arg_config.yaml: root must be a mapping, got {type(data).__name__}"
        )
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"agent_arg_config.yaml: missing or invalid section {name!r}")
    return section


def _parse_rag_context(data: object) -> RagContextConfig:
    section = data if isinstance(data, dict) else {}
    raw_max = section.get("max_chunks")
    max_chunks: int | None
    if raw_max is None:
        max_chunks = None
    else:
        max_chunks = int(raw_max)
        if max_chunks < 1:
            raise ValueError("agent_arg_config.yaml: rag_context.max_chunks must be >= 1")
    return RagContextConfig(max_chunks=max_chunks)


def _parse_pattern(raw: object) -> PatternConfig:
    data = raw if isinstance(raw, dict) else {}
    prompt_key = data.get("system_prompt", "default")
    if not isinstance(prompt_key, str) or not prompt_key.strip():
        prompt_key = "default"
    return PatternConfig(
        **{key: bool(data.get(key, False)) for key in _PATTERN_BOOL_FIELDS},
        system_prompt_key=prompt_key.strip(),
    )


@lru_cache(maxsize=8)
def load_agent_pattern_config(config_path: Path) -> AgentPatternConfig:
    """Load pattern definitions from an explicit YAML path."""
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    patterns_raw = _config_section(data, "patterns")
    patterns = {
        pattern_id: _parse_pattern(cfg)
        for pattern_id, cfg in patterns_raw.items()
    }
    rag_context_raw = data.get("rag_context") if isinstance(data, dict) else None
    return AgentPatternConfig(
        patterns=patterns,
        rag_context=_parse_rag_context(rag_context_raw),
    )


def get_agent_pattern_config(config_path: Path | None = None) -> AgentPatternConfig:
    path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
    return load_agent_pattern_config(path)


def get_pattern(
    pattern_id: str,
    config_path: Path | None = None,
) -> PatternConfig:
    """Return one named pattern; raises ``KeyError`` when unknown."""
    config = get_agent_pattern_config(config_path)
    try:
        return config.patterns[pattern_id]
    except KeyError as exc:
        known = ", ".join(sorted(config.patterns))
        raise KeyError(f"Unknown pattern {pattern_id!r}; known: {known}") from exc
