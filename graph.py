from langgraph.graph import StateGraph, START, END
from state import GraphState
from typing import Literal
from nodes import(
    supervisor_node,
    conversation_node,
    sql_node,
    web_research_node,
    visualization_node,
    rag_node
)
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

builder = StateGraph(GraphState)

builder.add_node("supervisor_node", supervisor_node)                       
builder.add_node("conversation_node", conversation_node)
builder.add_node("sql_node", sql_node)
builder.add_node("web_research_node", web_research_node)
builder.add_node("visualization_node", visualization_node)
builder.add_node("rag_node", rag_node)

def route_to_agent(state:GraphState) -> Literal["conversation_node", "sql_node", "web_research_node","visualization_node","rag_node"]:
    if state["next_agent"] == "conversation":
        return "conversation_node"
    elif state["next_agent"] == "web_research":
        return "web_research_node"
    elif state["next_agent"] == "visualization":
        return "visualization_node"
    elif state["next_agent"] == "rag":
        return "rag_node"
    elif state["next_agent"] == "sql":
        return "sql_node"
    else:
        return "conversation_node"

builder.add_edge(START, "supervisor_node")

builder.add_conditional_edges(
        "supervisor_node",
        route_to_agent
    )

builder.add_edge("conversation_node", END)
builder.add_edge("sql_node", END)
builder.add_edge("web_research_node", END)
builder.add_edge("visualization_node", END)
builder.add_edge("rag_node", END)

graph = builder.compile(checkpointer=memory)
