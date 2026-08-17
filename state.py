from typing import TypedDict, NotRequired ,Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class GraphState(TypedDict):
    next_agent: NotRequired[str]
    agent_results: NotRequired[dict]
    messages: Annotated[list[BaseMessage], add_messages]
    needs_visualization: NotRequired[bool]

