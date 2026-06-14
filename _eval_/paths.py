from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# Reuse the datasets already downloaded under the legacy eval/ tree so we do not
# duplicate multi-GB corpora. Override with EVAL_DATASETS_ROOT if they move.
DATASETS_ROOT = REPO_ROOT / "eval" / "datasets"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def dataset_dir(name: str) -> Path:
    """Return the root directory holding ``name``'s corpus/queries/qrels."""
    return DATASETS_ROOT / name


def results_dir() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT
