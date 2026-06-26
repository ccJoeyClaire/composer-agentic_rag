"""Load system prompts from ``system_prompt.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_PROMPT_KEY = "default"
_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.yaml"


@lru_cache(maxsize=4)
def load_system_prompts(prompt_path: Path = _PROMPT_PATH) -> dict[str, str]:
    """Load all named prompts from YAML."""
    with prompt_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"{prompt_path.name}: root must be a mapping, got {type(data).__name__}"
        )

    prompts: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        text = str(value).strip() if value is not None else ""
        if text:
            prompts[key] = text

    if _DEFAULT_PROMPT_KEY not in prompts:
        raise ValueError(f"{prompt_path.name}: missing required key {_DEFAULT_PROMPT_KEY!r}")
    return prompts


def get_system_prompt(key: str = _DEFAULT_PROMPT_KEY, *, prompt_path: Path | None = None) -> str:
    """Return one named system prompt; raises ``KeyError`` when unknown."""
    path = prompt_path if prompt_path is not None else _PROMPT_PATH
    prompts = load_system_prompts(path)
    try:
        return prompts[key]
    except KeyError as exc:
        known = ", ".join(sorted(prompts))
        raise KeyError(f"Unknown system prompt {key!r}; known: {known}") from exc


def default_system_prompt_key() -> str:
    return _DEFAULT_PROMPT_KEY
