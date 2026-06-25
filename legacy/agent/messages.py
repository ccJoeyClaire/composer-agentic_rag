"""LangChain message adapters for the agent graph.

Run (from repo root):
  python -m agent.messages
"""

import json
from typing import Any, List, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def messages_to_openai(messages: List[Union[BaseMessage, dict]]) -> List[dict]:
    """ state 中的 message 转化成 openai 格式 """
    result: List[dict] = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
            continue
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")
    return result


def openai_response_to_ai_message(response: Any) -> AIMessage:
    """解析 openai 的 response, 返回 AIMessage """
    tool_calls = None
    if getattr(response, "tool_calls", None): #相当于 response.tool_calls
        tool_calls = [
            {
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "id": tc.id,
            }
            for tc in response.tool_calls
        ]
    return AIMessage(content=response.content or "", tool_calls=tool_calls or [])


def _demo_main() -> None:
    """Inspectable demo: LangChain messages <-> OpenAI format."""
    from agent.state import DEFAULT_RAG_TOOL_NAME

    human = HumanMessage(content="What is the capital of France?")
    ai_tool = AIMessage(
        content="I will search the knowledge base.",
        tool_calls=[
            {
                "name": DEFAULT_RAG_TOOL_NAME,
                "args": {"query": "capital of France"},
                "id": "call_demo",
            }
        ],
    )
    tool = ToolMessage(content="Paris is the capital.", tool_call_id="call_demo")

    openai_msgs = messages_to_openai([human, ai_tool, tool])
    print("=== messages_to_openai ===")
    print(json.dumps(openai_msgs, indent=2, ensure_ascii=False))

    class _Fn:
        name = DEFAULT_RAG_TOOL_NAME
        arguments = json.dumps({"query": "capital of France"})

    class _Tc:
        id = "call_resp"
        function = _Fn()

    class _MockResponse:
        content = "The capital of France is Paris."
        tool_calls = None

    print("\n=== openai_response_to_ai_message ===")
    openai_response_to_ai_message(_MockResponse()).pretty_print()


if __name__ == "__main__":
    _demo_main()
