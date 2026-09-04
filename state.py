from typing import TypedDict, NotRequired ,Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class GraphState(TypedDict):
    next_agent: NotRequired[str]
    agent_results: NotRequired[dict]
    messages: Annotated[list[BaseMessage], add_messages]
    needs_visualization: NotRequired[bool]
    pending_write: NotRequired[dict]
    last_chart: NotRequired[dict]

class WebResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_context: NotRequired[str]
    research_error: NotRequired[str]
    web_next_step: NotRequired[str]
