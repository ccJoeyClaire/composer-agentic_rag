import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()


class LLMClient:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        timeout: int = None,
    ):
        self.model = model or os.environ.get("LLM_MODEL_ID")
        api_key = api_key or os.environ.get("LLM_API_KEY")
        base_url = base_url or os.environ.get("LLM_BASE_URL")
        if not all([self.model, api_key, base_url]):
            raise ValueError("必须同时拥有模型名称、API密钥和基础url")

        client_kwargs = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def _build_create_kwargs(
        self,
        messages: List[Dict],
        json_output: bool,
        tool_calls: bool,
        stream: bool,
        tools: List[Dict] | None,
        kwargs: dict,
    ) -> dict:
        if stream:
            raise NotImplementedError("stream 尚未实现")

        create_kwargs = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            **kwargs,
        }
        if json_output:
            create_kwargs["response_format"] = {"type": "json_object"}
        if tool_calls:
            if not tools:
                raise ValueError("tool_calls=True 但未提供 tools")
            create_kwargs["tools"] = tools
        return create_kwargs

    def request_llm(
        self,
        messages: List[Dict],
        json_output: bool = False,
        tool_calls: bool = False,
        stream: bool = False,
        tools: List[Dict] | None = None,
        **kwargs,
    ):
        """
        向 LLM 发送请求, message 格式类似:
            messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        """
        create_kwargs = self._build_create_kwargs(
            messages, json_output, tool_calls, stream, tools, kwargs
        )
        response = self.client.chat.completions.create(**create_kwargs)
        return response.choices[0].message

    async def arequest_llm(
        self,
        messages: List[Dict],
        json_output: bool = False,
        tool_calls: bool = False,
        stream: bool = False,
        tools: List[Dict] | None = None,
        **kwargs,
    ):
        create_kwargs = self._build_create_kwargs(
            messages, json_output, tool_calls, stream, tools, kwargs
        )
        response = await self.async_client.chat.completions.create(**create_kwargs)
        return response.choices[0].message


if __name__ == "__main__":
    import asyncio
    import json

    from tools.tool_box import ToolBox

    tool_box = ToolBox()
    client = LLMClient()

    SYSTEM_PROMPT = """
    你是一个可以调用工具的 AI 助手。
    需要计算时调用工具；信息足够时直接用自然语言回答用户。
    若需要调用工具，请先在正文简要说明推理，再发起 tool call。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "计算 x^2 + sin(x) + e^x 在(-3.5, 6) 上的积分",
        },
    ]

    llm_response = client.request_llm(
        messages,
        tool_calls=True,
        tools=tool_box.list_tools(),
    )
    content, tool_call = llm_response.content, llm_response.tool_calls

    if content:
        print("Reasoning:", content)
    else:
        print("Reasoning: (模型未返回正文，仅返回 tool call)")

    if tool_call:
        name = tool_call[0].function.name
        args = json.loads(tool_call[0].function.arguments)
        print(f"Tool call: {name}({args})")
        result = asyncio.run(tool_box.ainvoke(name, args))
        print("Observation:", result.output if result.error is None else result.error)
    else:
        print("Final answer:", content)
