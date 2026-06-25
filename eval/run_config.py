"""Eval run configuration and matrix builder.

An :class:`EvalRunConfig` fully identifies one run (index + infer unit).
:func:`build_run_matrix` generates the full 30-run grid from the blueprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.types import AGENT_ARMS, DIRECT_RAG_ARM, EVAL_PROFILES, EvalArm

# Qdrant collection prefix.  Override per-doc when adding new documents.
_DEFAULT_COLLECTION_PREFIX = "eval_codex"

# Separator used in run_id strings (must not appear in any component).
_RUN_ID_SEP = "__"


@dataclass(frozen=True)
class EvalRunConfig:
    """Fully-qualified identity for one indexing + inference run.

    Args:
        doc_slug:     Short stable identifier for the source document,
                      e.g. ``"codex"``.
        profile_id:   RAG index/retrieval profile, e.g. ``"baseline"``.
        arm:          Which inference arm to use (direct RAG or agent pattern).
        collection:   Qdrant collection name.  Defaults to
                      ``"{collection_prefix}_{profile_id}"``.
        collection_prefix: Used to build the default collection name.
    """

    doc_slug: str
    profile_id: str
    arm: EvalArm
    collection_prefix: str = _DEFAULT_COLLECTION_PREFIX
    collection: str = ""  # computed in __post_init__ when empty

    def __post_init__(self) -> None:
        # Frozen dataclass: use object.__setattr__ for computed fields.
        if not self.collection:
            object.__setattr__(
                self,
                "collection",
                f"{self.collection_prefix}_{self.profile_id}",
            )

    @property
    def run_id(self) -> str:
        """Canonical run identifier: ``{doc_slug}__{profile_id}__{arm}``."""
        return _RUN_ID_SEP.join([self.doc_slug, self.profile_id, self.arm])

    @property
    def is_direct_rag(self) -> bool:
        return self.arm == DIRECT_RAG_ARM

    @property
    def is_agent(self) -> bool:
        return self.arm in AGENT_ARMS


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def infer_output_path(config: EvalRunConfig) -> Path:
    """Where to write the raw infer result JSON for this run."""
    arm_dir = "retrieve" if config.is_direct_rag else "agent"
    return _RUNS_ROOT / arm_dir / f"{config.run_id}.json"


def checker_input_path(config: EvalRunConfig) -> Path:
    """Where to write the RAGChecker input JSON for this run."""
    return _RUNS_ROOT / "checking_inputs" / f"{config.run_id}.json"


def checker_output_path(config: EvalRunConfig) -> Path:
    """Where to write the RAGChecker output JSON for this run."""
    return _RUNS_ROOT / "checking_outputs" / f"{config.run_id}.json"


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------


def build_run_matrix(
    doc_slug: str,
    *,
    profiles: tuple[str, ...] = EVAL_PROFILES,
    arms: tuple[EvalArm, ...] = (DIRECT_RAG_ARM,) + AGENT_ARMS,
    collection_prefix: str = _DEFAULT_COLLECTION_PREFIX,
) -> list[EvalRunConfig]:
    """Generate the full eval matrix for one document.

    Produces one :class:`EvalRunConfig` per (profile × arm) combination.
    Default matrix = 6 profiles × 5 arms = 30 runs (blueprint §4.4).

    Args:
        doc_slug:           Short stable document identifier.
        profiles:           RAG profile IDs to include.
        arms:               Inference arms to include.
        collection_prefix:  Qdrant collection name prefix.

    Returns:
        Flat list of run configs in (profile, arm) order.
    """
    return [
        EvalRunConfig(
            doc_slug=doc_slug,
            profile_id=profile_id,
            arm=arm,
            collection_prefix=collection_prefix,
        )
        for profile_id in profiles
        for arm in arms
    ]


def index_configs_for_matrix(matrix: list[EvalRunConfig]) -> list[EvalRunConfig]:
    """Deduplicated index configs: one per unique (doc_slug, profile_id).

    Each profile needs to be indexed exactly once, regardless of how many
    arms share it.  The returned configs all use ``direct_rag`` as a sentinel;
    only ``doc_slug``, ``profile_id``, and ``collection`` matter for indexing.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[EvalRunConfig] = []
    for cfg in matrix:
        key = (cfg.doc_slug, cfg.profile_id)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique
