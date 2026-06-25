"""Seed the canonical system prompt — callers must not supply their own."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, RemoveMessage, SystemMessage

from agent.core.state import AgentState
from agent.prompt.load import get_system_prompt


def seed_system_prompt_node(
    state: AgentState,
    *,
    system_prompt_key: str,
) -> dict[str, object]:
    """Prepend the YAML system prompt and drop any caller-supplied ``SystemMessage``."""
    prompt = get_system_prompt(system_prompt_key)
    messages = list(state["messages"])

    if (
        messages
        and isinstance(messages[0], SystemMessage)
        and messages[0].content == prompt
    ):
        return {}

    non_system = [message for message in messages if not isinstance(message, SystemMessage)]
    updates: list[BaseMessage | RemoveMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage) and message.id:
            updates.append(RemoveMessage(id=message.id))

    updates.append(SystemMessage(content=prompt))
    updates.extend(non_system)
    return {"messages": updates}
