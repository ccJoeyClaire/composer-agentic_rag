from __future__ import annotations

from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent


def dataset_dir(name: str) -> Path:
    return EVAL_ROOT / "datasets" / name
