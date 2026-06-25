from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
RESULTS_ROOT = PACKAGE_ROOT / "results"

def datasets_root() -> Path:
    """Return the root directory for BEIR dataset files.

    Resolution order:
    1. ``EVAL_DATASETS_ROOT`` env var (absolute or repo-relative path).
    2. ``_eval_/datasets`` (created on demand for new downloads).
    """
    if env_root := os.environ.get("EVAL_DATASETS_ROOT"):
        path = Path(env_root)
        return path if path.is_absolute() else REPO_ROOT / path

    return PACKAGE_ROOT / "datasets"


def dataset_dir(name: str) -> Path:
    """Return the root directory holding ``name``'s corpus/queries/qrels."""
    return datasets_root() / name


def results_dir() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT
