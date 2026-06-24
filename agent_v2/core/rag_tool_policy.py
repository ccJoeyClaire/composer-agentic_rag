"""Agent policy for ``RAG_search_tool`` — query-only vs full override schema."""

from __future__ import annotations

import copy
from typing import Any

from rag.profile_schema import SEARCH_PROFILE_KEYS

RAG_SEARCH_QUERY_ARG = "query"


def rag_search_args_query_only(args: dict[str, Any]) -> dict[str, Any]:
    """Keep only ``query``; drop profile override keys (deployment defaults apply)."""
    return {RAG_SEARCH_QUERY_ARG: str(args.get(RAG_SEARCH_QUERY_ARG, ""))}


def restrict_rag_search_tool_schema(
    schema: dict[str, Any],
    *,
    rag_tool_name: str,
) -> dict[str, Any]:
    """Return a copy of *schema* with only the ``query`` parameter exposed."""
    fn = schema.get("function") or {}
    if fn.get("name") != rag_tool_name:
        return schema

    restricted = copy.deepcopy(schema)
    params = restricted.setdefault("function", {}).setdefault("parameters", {})
    properties = params.get("properties") or {}
    params["properties"] = {
        key: value
        for key, value in properties.items()
        if key == RAG_SEARCH_QUERY_ARG
    }
    required = params.get("required") or []
    params["required"] = [key for key in required if key == RAG_SEARCH_QUERY_ARG]
    return restricted


def rag_search_profile_arg_keys() -> frozenset[str]:
    """Keys the LLM may use to override search profile when router is enabled."""
    return frozenset(SEARCH_PROFILE_KEYS)
