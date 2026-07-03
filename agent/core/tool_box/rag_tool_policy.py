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


if __name__ == "__main__":
    import json

    from agent.core.tool_box.constants import DEFAULT_RAG_TOOL_NAME

    sample_schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": DEFAULT_RAG_TOOL_NAME,
            "description": "Search the knowledge base and return relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    RAG_SEARCH_QUERY_ARG: {
                        "type": "string",
                        "description": "检索问题",
                    },
                    "use_hyde": {
                        "type": "boolean",
                        "description": "是否用 HyDE 改写查询向量",
                    },
                    "recall_n": {
                        "type": "integer",
                        "description": "rerank 前的向量召回条数",
                    },
                },
                "required": [RAG_SEARCH_QUERY_ARG],
            },
        },
    }

    restricted = restrict_rag_search_tool_schema(
        sample_schema,
        rag_tool_name=DEFAULT_RAG_TOOL_NAME,
    )

    print("=== before (full override schema) ===")
    print(json.dumps(sample_schema, indent=2, ensure_ascii=False))
    print("\n=== after restrict_rag_search_tool_schema (query only) ===")
    print(json.dumps(restricted, indent=2, ensure_ascii=False))
    print("\n=== property keys removed ===")
    before_keys = set(
        sample_schema["function"]["parameters"]["properties"].keys()
    )
    after_keys = set(restricted["function"]["parameters"]["properties"].keys())
    print(sorted(before_keys - after_keys))

# python -m agent.core.tool_box.rag_tool_policy

# === after restrict_rag_search_tool_schema (query only) ===
# {
#   "type": "function",
#   "function": {
#     "name": "RAG_search_tool",
#     "description": "Search the knowledge base and return relevant context.",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "query": {
#           "type": "string",
#           "description": "检索问题"
#         }
#       },
#       "required": [
#         "query"
#       ]
#     }
#   }
# }

# === property keys removed ===
