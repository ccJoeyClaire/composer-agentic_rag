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
class PatternConfig:
    """Capability toggles for one named reflection pattern."""

    enable_retrieval_gate: bool
    enable_rag_profile_router: bool
    enable_human_feedback: bool


@dataclass(frozen=True)
class AgentPatternConfig:
    """Root document loaded from ``agent_arg_config.yaml``."""

    patterns: Dict[str, PatternConfig]


def _config_section(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError(
            f"agent_arg_config.yaml: root must be a mapping, got {type(data).__name__}"
        )
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"agent_arg_config.yaml: missing or invalid section {name!r}")
    return section


def _parse_pattern(raw: object) -> PatternConfig:
    data = raw if isinstance(raw, dict) else {}
    return PatternConfig(
        **{key: bool(data.get(key, False)) for key in _PATTERN_BOOL_FIELDS}
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
    return AgentPatternConfig(patterns=patterns)


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
