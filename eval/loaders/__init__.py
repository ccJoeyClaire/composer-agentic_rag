"""Eval dataset loaders — canonical ``EvalDocument`` + adapter registry."""

from eval.loaders.load import load_dataset_config, load_documents
from eval.loaders.schema import DatasetConfig, EvalDocument

# Backward-compatible names used by older eval code.
from eval.loaders.adapters.native import parse_native_record, resolve_repo_path

__all__ = [
    "DatasetConfig",
    "EvalDocument",
    "load_dataset_config",
    "load_documents",
    "parse_native_record",
    "resolve_repo_path",
]
