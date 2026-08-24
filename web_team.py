from typing import Literal

from langgraph.graph import StateGraph, START,END
from langchain_core.messages import AIMessage
from state import WebResearchState
from nodes import (
    web_research_supervisor_node,
    researcher_node,
    report_writer_node
)

builder = StateGraph(WebResearchState)

builder.add_node(
    "web_research_supervisor",
    web_research_supervisor_node
)

builder.add_node(
    "researcher",
    researcher_node
)

builder.add_node(
    "report_writer",
    report_writer_node
)

builder.add_edge(
    START,
    "web_research_supervisor"
)

def route_web_team(
        state: WebResearchState
) -> Literal["researcher", "report_writer", "fallback"]:

    return state["web_next_step"]

def web_fallback_node(state: WebResearchState):
    error = state.get(
        "research_error",
        "the available web research was insufficient to answer the request."
    )

    if not error:
        error = (
            "The available web research was insufficient "
            "to answer that request."
        )

    return {
        "messages": [
            AIMessage(content=error)
        ]
    }

builder.add_node(
    "fallback",
    web_fallback_node
)

builder.add_conditional_edges(
    "web_research_supervisor",
    route_web_team
)

builder.add_edge(
    "researcher",
    "web_research_supervisor"
)

builder.add_edge(
    "report_writer",
    END
)

builder.add_edge(
    "fallback",
    END
)

web_team_graph = builder.compile()