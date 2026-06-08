from typing import NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, Dict, List, TypedDict

# Reflection metadata field names live in agent.metadata_schema.AgentMetadata.


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    metadata: NotRequired[Dict]
    tool_calls: NotRequired[Dict]
    error: NotRequired[str]
