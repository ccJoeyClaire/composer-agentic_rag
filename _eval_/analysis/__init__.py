"""Post-hoc analysis for RAG eval run JSON artifacts and live replay tools.

Subpackages::

    visual/   — notebooks and charts over ``_eval_/results/*.json``
    replay/   — inspect Qdrant collections and replay query pipelines
"""

from _eval_.analysis.compare import (
    PairedTTestResult,
    WinrateResult,
    delta_per_query,
    mean_delta,
    mean_delta_table,
    paired_ttest,
    winrate,
    winrate_matrix,
)
from _eval_.analysis.load_results import (
    ProfileResult,
    QueryScore,
    RunConfigSnapshot,
    RunResult,
    common_query_ids,
    load_run_result,
    metric_names,
    per_query_metrics,
    profile_map,
)

__all__ = [
    "PairedTTestResult",
    "ProfileResult",
    "QueryScore",
    "RunConfigSnapshot",
    "RunResult",
    "WinrateResult",
    "common_query_ids",
    "delta_per_query",
    "load_run_result",
    "mean_delta",
    "mean_delta_table",
    "metric_names",
    "paired_ttest",
    "per_query_metrics",
    "profile_map",
    "winrate",
    "winrate_matrix",
]
