from state import GraphState
from langchain_core.messages import SystemMessage
from llm import create_llm
from langchain_core.messages import AIMessage


llm = create_llm()

def supervisor_node(state: GraphState):
 latest_message = state['messages'][-1]

 system_prompt = (
        "Classify the user's request into exactly one category: "
        "conversation, sql, web_research ,visualization, or rag. "
        "Return only the category name."
    )

 
 response = llm.invoke([
   SystemMessage(content=system_prompt),
   latest_message]
 )

 route = response.content.lower().strip()


 allowed_routes = {
    "conversation",
    "sql",
    "web_research",
    "visualization",
    "rag",
 }
 if route not in allowed_routes:
    route = "conversation"

 return {"next_agent": route}

def conversation_node(state: GraphState):
    conversation_message = [SystemMessage(content= "You are the general conversation agent. "
                "Handle greetings, explanations, and general questions. "
                "Do not pretend to query databases, browse the web, "
                "retrieve internal documents, or create charts.")] + state["messages"]
    response = llm.invoke(conversation_message)
    return {"messages": [response]}

def sql_node(state: GraphState):
   response = AIMessage(content="SQL placeholder")
   return {"messages": [
      response
   ]}

def web_research_node(state: GraphState):
   response = AIMessage(content="web_research placeholder")
   return {"messages": [
      response
   ]}

def visualization_node(state: GraphState):
   response = AIMessage(content="Visualization placeholder")
   return {"messages": [
      response
   ]}

def rag_node(state: GraphState):
   response = AIMessage(content="RAG placeholder")
   return {"messages": [
      response
   ]}