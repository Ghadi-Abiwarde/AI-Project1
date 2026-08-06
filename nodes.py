from state import GraphState
from langchain_core.messages import SystemMessage
from llm import create_llm
from langchain_core.messages import AIMessage
from database import execute_query
import json

llm = create_llm()

def supervisor_node(state: GraphState):
 latest_message = state['messages'][-1]

 system_prompt = (
        "Classify the user's request into exactly one category: "
        "conversation, sql, web_research ,visualization, or rag. "
        "Return valid JSON using exactly this structure: "
        '{"next_agent": "category_name"}. '
        "Do not include explanations or markdown."
        "Use web_research when the request requires current or external information from the internet."
        "Use rag when the request refers to internal documents, company policies, handbooks, manuals, uploaded files, or indexed knowledge."
        )

 
 response = llm.invoke([
   SystemMessage(content=system_prompt),
   latest_message]
 )


 try:
  decision = json.loads(response.content)
  route = decision["next_agent"].strip().lower()
 except (json.JSONDecodeError, KeyError, AttributeError):
  route = "conversation"

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
    conversation_message = [SystemMessage(content= "You are the general conversation agent."
                                          "Handle general conversational interactions, explanations, educational questions, and conceptual discussions."
                                          "Structure your answers to be professional, friendly, clear, and concise. You should sound trustworthy."
                                          "Recognize when another specialized agent is better suited for the request. Do not fabricate capabilities you do not have. Clearly explain which specialized agent should handle the request."
                                          "Maintain continuity across the conversation by incorporating relevant information from previous messages whenever it improves the response."
                                          )] + state["messages"]
    response = llm.invoke(conversation_message)
    return {"messages": [response]}






def sql_node(state: GraphState):
   latest_message = state["messages"][-1]
   sql_prompt = """Role:
                  You are the SQL agent. Convert the user's question into one PostgreSQL SELECT query.
                  Database schema:
                  customers(
                  customer_id,
                  name,
                  email,
                  phone_number,
                  address,
                  join_date)

                  suppliers(
                  supplier_id,
                  supplier_name,
                  email,
                  phone_number,
                  address)

                  products(
                  product_id,
                  product_name,
                  supplier_id,
                  category,
                  current_stock,
                  price)

                  orders(
                  order_id,
                  customer_id,
                  order_date,
                  total_amount)

                  order_items(
                  order_item_id,
                  order_id,
                  product_id,
                  quantity,
                  unit_price)

                  Relationships:
                  products.supplier_id references suppliers.supplier_id.
                  orders.customer_id references customers.customer_id.
                  order_items.order_id references orders.order_id.
                  order_items.product_id references products.product_id.

                  Rules:
                  Generate exactly one PostgreSQL SELECT query.
                  Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
                  Use only the provided tables and columns.
                  Return valid JSON using exactly this structure:
                  {"query": "SELECT ..."}
                  Do not include explanations or Markdown.
                              """
   

   response = llm.invoke([
     SystemMessage(content=sql_prompt), latest_message
   ])

   try:
     decision = json.loads(response.content)
     query = decision["query"].strip()
   except (json.JSONDecodeError, KeyError, AttributeError):
      return {
    "messages": [
        AIMessage(content="I was unable to generate a valid SQL query.")
    ]
}
   if not query.lower().startswith("select"):
      return {
        "messages": [
            AIMessage(content="Only SELECT queries are allowed.")
        ]
    }
   
   try:
       results = execute_query(query)
   except Exception as error:
     return {
        "messages": [
            AIMessage(content=f"The database query could not be executed: {error}")
        ]
    }   

     
  
   return {
    "messages": [
        AIMessage(content=str(results))
    ]
}







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