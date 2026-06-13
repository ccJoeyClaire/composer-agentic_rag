from __future__ import annotations

from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent


SMOKE_DATASET_NAME = "smoke"


def dataset_dir(name: str) -> Path:
    """Return ``eval/datasets/<name>/`` (smoke lives here too)."""
    return EVAL_ROOT / "datasets" / name


def smoke_dir() -> Path:
    return dataset_dir(SMOKE_DATASET_NAME)
