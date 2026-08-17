from state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from llm import create_llm
from database import execute_query, get_database_schema
from rag.retriever import retrieve_documents
from web_search import search_web
from visualization import create_chart, validate_chart_data
from decimal import Decimal
import json

llm = create_llm()

def supervisor_node(state: GraphState):
 latest_message = state['messages'][-1]

 system_prompt = """
You are the supervisor agent.

Classify the user's request into exactly one category:
- conversation
- sql
- web_research
- visualization
- rag

Return valid JSON using exactly this structure:
{"next_agent": "category_name", "needs_visualization": false}

Routing rules:

1. conversation
Use conversation for greetings, general knowledge, explanations, writing, and requests that do not require another specialized agent.

2. sql
Use sql when the request requires retrieving or analyzing data from the PostgreSQL database.

3. web_research
Use web_research when the request requires current or external information from the internet.

4. rag
Use rag when the request refers to internal documents, company policies, handbooks, manuals, uploaded files, or indexed knowledge.
Use rag for questions about company-specific employee policies, workplace rules,
leave, probation, remote work, conduct, equipment, HR procedures, or internal company practices,
even if the user does not explicitly mention "handbook", "policy", or "internal documents".

Contrasting examples:
User: What happens if an employee is repeatedly late?
Output: {"next_agent": "rag", "needs_visualization": false}

User: Why is punctuality important at work?
Output: {"next_agent": "conversation", "needs_visualization": false}


5. visualization
Use visualization when the user provides the data needed to create a chart directly.

Visualization and SQL rules:

If the user provides the chart data directly:
- next_agent = visualization
- needs_visualization = false

If the requested visualization requires retrieving data from the PostgreSQL database:
- next_agent = sql
- needs_visualization = true

For all other requests:
- needs_visualization = false

Examples:

User: Create a pie chart from A=40, B=35, C=25.
Output: {"next_agent": "visualization", "needs_visualization": false}

User: Create a bar chart showing total spending by customer.
Output: {"next_agent": "sql", "needs_visualization": true}

User: Which customer spent the most money?
Output: {"next_agent": "sql", "needs_visualization": false}

Do not include explanations or Markdown.
Return only the JSON object.
"""

 
 response = llm.invoke([
   SystemMessage(content=system_prompt),
   latest_message]
 )


 try:
   decision = json.loads(response.content)

   route = decision["next_agent"].strip().lower()
   needs_visualization = decision["needs_visualization"]
 except (json.JSONDecodeError, KeyError, AttributeError):
   route = "conversation"
   needs_visualization = False

 allowed_routes = {
    "conversation",
    "sql",
    "web_research",
    "visualization",
    "rag",
 }
 if route not in allowed_routes:
    route = "conversation"
    needs_visualization = False

 if not isinstance(needs_visualization, bool):
    needs_visualization = False      

 return {"next_agent": route,
         "needs_visualization": needs_visualization
         }







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
    database_schema = get_database_schema()

    sql_prompt = f"""Role:
You are the SQL agent. Convert the user's question into one PostgreSQL SELECT query.

Database schema:
{database_schema}

Relationships:
products.supplier_id references suppliers.supplier_id.
orders.customer_id references customers.customer_id.
order_items.order_id references orders.order_id.
order_items.product_id references products.product_id.

Rules:
Generate exactly one PostgreSQL SELECT query.
Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
Use only the provided tables and columns.
When the question refers to a specific named entity such as a customer, product, or supplier,
include the identifying field in the query result.
For aggregate questions about a specific named entity, the query must only return a row if that entity actually exists.

Do not use aggregate patterns that return a default zero row when the named entity does not exist.
For example, when counting orders for a specific customer:
- filter by the customer in the customers table
- group by that customer
- if the customer does not exist, return no rows

Return valid JSON using exactly this structure:
{{"query": "SELECT ..."}}

Do not include explanations or Markdown.

When an aggregate, calculation, ranking, or comparison is used to answer the question,
include the calculated value in the SELECT output with a descriptive alias.

All numeric database values must remain numeric in the SELECT output.
Do not format numeric values for display inside SQL.
Do not use TO_CHAR for numeric formatting.
Do not add currency symbols.
Do not convert numeric values to text.
Human-readable formatting will be handled after the query executes.
"""

    response = llm.invoke([
        SystemMessage(content=sql_prompt),
        latest_message
    ])

    try:
        decision = json.loads(response.content)
        query = decision["query"].strip()

    except (json.JSONDecodeError, KeyError, AttributeError):
        return {
            "messages": [
                AIMessage(
                    content="I was unable to generate a valid SQL query."
                )
            ]
        }

    # If GPT-OSS formats numeric values using TO_CHAR,
    # regenerate the query once with stricter instructions.
    if "to_char(" in query.lower():
        correction_prompt = f"""
The following PostgreSQL query incorrectly formats numeric values as text:

{query}

Regenerate the query while following these rules:
- Keep all numeric database values numeric.
- Do not use TO_CHAR.
- Do not add currency symbols.
- Do not convert numeric values to text.
- Preserve the meaning of the original query.
- Return valid JSON using exactly this structure:
{{"query": "SELECT ..."}}
- Return no explanations or Markdown.
"""

        retry_response = llm.invoke([
            SystemMessage(content=correction_prompt)
        ])

        try:
            retry_decision = json.loads(retry_response.content)
            query = retry_decision["query"].strip()

        except (json.JSONDecodeError, KeyError, AttributeError):
            return {
                "messages": [
                    AIMessage(
                        content="I was unable to generate a valid database query."
                    )
                ]
            }

    # If the retry still contains TO_CHAR, reject it.
    if "to_char(" in query.lower():
        return {
            "messages": [
                AIMessage(
                    content="The generated database query contained unsupported numeric formatting."
                )
            ]
        }

    # Safety check: SELECT queries only.
    if not query.lower().startswith("select"):
        return {
            "messages": [
                AIMessage(
                    content="Only SELECT queries are allowed."
                )
            ]
        }

    try:
        results = execute_query(query)

        if not results:
            return {
                "messages": [
                    AIMessage(
                        content="No matching information was found in the database."
                    )
                ],
                "agent_results": {
                    "sql": []
                }
            }

        natural_language_prompt = (
            "Role: You translate database results into natural language. "
            "Answer the user's question using only the provided database results. "
            "Do not fabricate data. "
            "If the results do not contain enough information, say so clearly. "
            "When presenting monetary values, format them clearly as currency "
            "using a dollar sign and appropriate thousands separators when applicable. "
            "Do not include Markdown."
        )

        final_response = llm.invoke([
            SystemMessage(content=natural_language_prompt),
            HumanMessage(
                content=(
                    f"User question: {latest_message.content}\n"
                    f"Database results: {results}"
                )
            )
        ])

    except Exception as error:
        print("SQL execution error:", error)

        return {
            "messages": [
                AIMessage(
                    content=f"I could not safely execute that database query: {error}"
                )
            ]
        }

    return {
        "messages": [
            final_response
        ],
        "agent_results": {
            "sql": results
        }
    }










def web_research_node(state: GraphState):
    latest_message = state["messages"][-1]
    try:
      results = search_web(latest_message.content)
    except Exception as error:
       print("Web search error:", error)
       return {
          "messages":[
             AIMessage(content="I couldn't retrieve web search results for that request."
           ) 
           ]
            }

    context = ""

    for result in results:
        context += (
            f"Title: {result['title']}\n"
            f"Link: {result['link']}\n"
            f"Snippet: {result['snippet']}\n\n"
        )
        sitelinks = result.get("sitelinks", [])

        for sitelink in sitelinks:
           context += (
              f" Sitelink title: {sitelink.get('title', '')}\n"
              f" Sitelink link: {sitelink.get('link', '')}\n"
              f" Sitelink snippet: {sitelink.get('snippet', '')}\n"
           )
    context += "\n"


    web_prompt = """Role:
You are the web research agent.

Responsibilities:
Answer the user's question using the provided web search results.

Rules:
Use only the provided search results.
Do not fabricate facts.
If the results are insufficient, say so clearly.
Mention relevant sources when useful.
Prefer official documentation, primary sources, and authoritative organizations over blogs, videos, forums, or social media when the same information is available from a primary source.
Present the source as the following examples:
-"According to the official Python 3.14 documentation, major changes include ..."
-"Sources: Python documentation, Python Developer’s Guide"
Do not attribute a claim to a source unless that claim is supported by that specific search result.
When the user asks for the current or latest version of a product, clearly identify the newest stable release first. Distinguish stable releases from beta, preview, or prerelease versions.
Be concise and clear.
For questions asking for the latest, newest, or current version:
- Use only the provided search results.
- Prefer official or primary sources.
- Use sitelinks when they provide clearer version information.
- Do not rely on prior knowledge.
- If the search results do not clearly establish the current version, say that the information is insufficient.
"""

    response = llm.invoke([
        SystemMessage(content=web_prompt),
        HumanMessage(
            content=f"""
Search results:
{context}

User question:
{latest_message.content}
"""
        )
    ])

    return {
        "messages": [response]
    }










   
def visualization_node(state: GraphState):
   latest_message = state["messages"][-1]
   
   visualization_prompt = f"""Role:
   You are the Visualization agent.
   Extract the chart type, labels, and numeric values needed to create a chart from the user's request.
   Rules:
   Generate chart type, labels and values.
   Return valid JSON using exactly this structure:
   {{"chart_type": "pie",
   "labels":["A", "B", "C"],
   "values":[40, 35,25]}}
   Do not include explanations or Markdown.
"""
   response = llm.invoke([
   SystemMessage(content=visualization_prompt),latest_message
])   
   sql_results = state.get("agent_results",{}).get("sql")
   


  
   if sql_results:
      first_row = sql_results[0]
      columns = list(first_row.keys())


      if len(columns)<2:
                return {
            "messages": [
               AIMessage(
                  content = "The database results do not contain enough data to create a chart."
               )
            ]
         }
      label_column = None
      value_column = None
      
      for column, value in first_row.items():

         if isinstance(value, str) and label_column is None:
            label_column = column

         elif isinstance(value, (int, float, Decimal)) and not column.lower().endswith("_id") and value_column is None:
            value_column = column

      if label_column is None or value_column is None:
            return {
               "messages": [
                     AIMessage(
                        content="The database results do not contain suitable data for a chart."
                     )
               ]
    }  
      labels = [
            row[label_column]
            for row in sql_results
         ]

      values = [
            float(row[value_column])
            for row in sql_results
         ]


      if "pie" in latest_message.content.lower():
         chart_type = "pie"
      else:
         chart_type = "bar"


   else:
    try:
      decision = json.loads(response.content)
      chart_type = decision["chart_type"].strip().lower()
      labels = decision["labels"]
      values = decision["values"]
    except (json.JSONDecodeError, KeyError, AttributeError):
      return{
         "messages":[
            AIMessage(content="I was unable to generate the chart.")
         ]
      }  

   validation_error = validate_chart_data(
    chart_type,
    labels,
    values
)

   if validation_error:
      return {
         "messages": [
               AIMessage(content=validation_error)
         ]
      }
   
  
   try:
      create_chart(chart_type, labels, values)

   except Exception as error:
      print("Visualization error:", error)

      return {
         "messages":[
            AIMessage(content="The chart could not be created.")
         ]
      }
   return{
      "messages": [
         AIMessage(content="Chart created successfully.")
      ]
   }

   
   


  
   





def rag_node(state: GraphState):
   latest_message = state["messages"][-1]
   try:
      documents = retrieve_documents(latest_message.content)

   except Exception as error:
      print("RAG retrieval error:", error) 

      return {
         "messages": [
            AIMessage(
               content="I couldn't retrieve information from the internal documents."
            )
         ]
      }

   if not documents:
      return {
         "messages": [
            AIMessage(
               content="The information is not available in the internal documents."
            )
         ]
      }

   context = "\n\n".join(
    document.page_content
    for document in documents
)
   rag_prompt = """Role:
You are the internal knowledge RAG agent.

Responsibilities:
Answer questions using only the retrieved internal document context.

Grounding rules:
- Answer using only information explicitly stated in the provided context.
- Do not use general knowledge to fill missing information.
- Do not infer company rules, procedures, benefits, or policies that are not explicitly stated.
- If the context only partially answers the question, provide only the supported information and clearly state that the remaining information is not available.
- If the context does not answer the question, respond exactly:
  "The information is not available in the internal documents."
- Answer clearly and concisely.
"""


   response = llm.invoke([
      SystemMessage(content=rag_prompt),
      HumanMessage(
         content=f"""
   Retrieved context:
   {context}

   User question:
   {latest_message.content}
   """
      )
   ])
   return {"messages": [
      response
   ]}