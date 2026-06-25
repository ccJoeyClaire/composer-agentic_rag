"""Re-export HyDE log helpers (implementation lives in :mod:`rag.query_transformer.hyde`)."""

from rag.query_transformer.hyde import HydeLogEntry, append_hyde_log_entry, hyde_log_path

__all__ = ["HydeLogEntry", "append_hyde_log_entry", "hyde_log_path"]
