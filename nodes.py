from state import GraphState, WebResearchState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from llm import create_llm
from database import execute_query, get_database_schema
from rag.retriever import retrieve_documents
from web_search import search_web
from visualization import create_chart, validate_chart_data
from decimal import Decimal
from database import (
    execute_query,
    execute_write,
    validate_write_query,
    get_database_schema
)
import json
import logging

logger = logging.getLogger(__name__)


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
You are the SQL agent.

Convert the user's request into exactly one PostgreSQL database operation.

Database schema:
{database_schema}

Relationships:
products.supplier_id references suppliers.supplier_id.
orders.customer_id references customers.customer_id.
order_items.order_id references orders.order_id.
order_items.product_id references products.product_id.

Allowed operations:
- SELECT
- INSERT
- UPDATE
- DELETE

Forbidden operations:
- DROP
- ALTER
- TRUNCATE
- CREATE
- GRANT
- REVOKE

Rules:
Generate exactly one PostgreSQL statement.
Use only the provided tables and columns.
Do not generate multiple SQL statements.

For UPDATE and DELETE:
- Always include a WHERE clause.
- The WHERE clause must identify the records the user explicitly requested to modify.
- Never update or delete every row in a table.

For SELECT:
- When the question refers to a specific named entity such as a customer,
  product, or supplier, include the identifying field in the query result.
- For aggregate questions about a specific named entity, only return a row
  if that entity actually exists.
- Do not use aggregate patterns that return a default zero row when the
  named entity does not exist.
- Keep numeric values numeric.
- Do not use TO_CHAR for numeric formatting.
- Do not add currency symbols.
- Do not convert numeric values to text.

Return valid JSON using exactly this structure:
{{"operation": "select", "query": "SELECT ..."}}

The value of "operation" must be exactly one of:
- select
- insert
- update
- delete

The operation value must match the actual SQL statement.

Do not include explanations or Markdown.
"""

    response = llm.invoke([
        SystemMessage(content=sql_prompt),
        latest_message
    ])

    try:
        decision = json.loads(response.content)

        operation = decision["operation"].strip().lower()
        query = decision["query"].strip()

    except (
        json.JSONDecodeError,
        KeyError,
        AttributeError
    ):
        return {
            "messages": [
                AIMessage(
                    content="I was unable to generate a valid database query."
                )
            ]
        }

    allowed_operations = {
        "select",
        "insert",
        "update",
        "delete"
    }

    if operation not in allowed_operations:
        return {
            "messages": [
                AIMessage(
                    content="The requested database operation is not supported."
                )
            ]
        }

    # Verify that the declared operation matches the generated SQL.
    actual_operation = query.split(maxsplit=1)[0].lower()

    if actual_operation != operation:
        return {
            "messages": [
                AIMessage(
                    content="The generated database operation did not pass validation."
                )
            ]
        }

    # -------------------------
    # SELECT path
    # -------------------------
    if operation == "select":

        # Preserve the existing numeric-formatting safeguard.
        if "to_char(" in query.lower():
            correction_prompt = f"""
The following PostgreSQL SELECT query incorrectly formats numeric values as text:

{query}

Regenerate the query while following these rules:
- Keep all numeric database values numeric.
- Do not use TO_CHAR.
- Do not add currency symbols.
- Do not convert numeric values to text.
- Preserve the meaning of the original query.

Return valid JSON using exactly this structure:
{{"operation": "select", "query": "SELECT ..."}}

Do not include explanations or Markdown.
"""

            retry_response = llm.invoke([
                SystemMessage(content=correction_prompt)
            ])

            try:
                retry_decision = json.loads(
                    retry_response.content
                )

                retry_operation = (
                    retry_decision["operation"]
                    .strip()
                    .lower()
                )

                query = (
                    retry_decision["query"]
                    .strip()
                )

            except (
                json.JSONDecodeError,
                KeyError,
                AttributeError
            ):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I was unable to generate "
                                "a valid database query."
                            )
                        )
                    ]
                }

            if retry_operation != "select":
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "The regenerated database query "
                                "did not pass validation."
                            )
                        )
                    ]
                }

        if "to_char(" in query.lower():
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "The generated database query contained "
                            "unsupported numeric formatting."
                        )
                    )
                ]
            }

        try:
            results = execute_query(query)

            if not results:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "No matching information was found "
                                "in the database."
                            )
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
                SystemMessage(
                    content=natural_language_prompt
                ),
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
                        content=(
                            "I could not safely execute that "
                            f"database query: {error}"
                        )
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

    # -------------------------
    # WRITE path
    # -------------------------
    validation_error = validate_write_query(query)
    
    if validation_error:
        return {
            "messages": [
                AIMessage(
                    content=validation_error
                )
            ]
        }

#UPDATE nad DELETE require confirmation
    if operation in {"update", "delete"}:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"This operation will {operation} database records. "
                        "Please confirm by replying yes or no"
                    )
                )
            ],
            "pending_write": {
                "operation": operation,
                "query": query
            }
        }
    

    try:
        affected_rows = execute_write(query)

    except Exception as error:
        print("SQL write error:", error)

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I could not safely execute that "
                        f"database write operation: {error}"
                    )
                )
            ]
        }

    if operation == "insert":
        message = (
            f"Insert completed successfully. "
            f"{affected_rows} row(s) affected."
        )

    elif operation == "update":
        message = (
            f"Update completed successfully. "
            f"{affected_rows} row(s) affected."
        )

    else:
        message = (
            f"Delete completed successfully. "
            f"{affected_rows} row(s) affected."
        )

    return {
        "messages": [
            AIMessage(content=message)
        ]
    }






def confirm_write_node(state: GraphState):
    latest_message = state["messages"][-1].content.strip().lower()
    pending_write = state.get("pending_write")

    if not pending_write:
        return{
            "messages": [
                AIMessage(content="There is no pending database option to confirm.")
            ]
        }
    if latest_message in {"yes", "confirm", "yes confirm"}:
        try:
            affected_rows = execute_write(
                pending_write["query"]
            )

            if affected_rows == 0:
                return {
                    "messages": [
                        AIMessage(
                            content='No matching records were found, so no changes were made.'
                        )
                    ],
                    "pending_write": {}
                }

            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"{pending_write['operation'].capitalize()}"
                            f"completed successfully. "
                            f"{affected_rows} row(s) affected."
                        )
                    )
                ],
                "pending_write": {}
            }

        except Exception as error:
            return {
                "messages": [
                    AIMessage(
                        content=f"The database operation failed: {error}"
                    )
                ],
                "pending_write": {}
            }

    if latest_message in {"no", "cancel", "no cancel"}:
        return {
            "messages": [
                AIMessage(
                    content="The database operation was cancelled."
                )
            ],
            "pending_write": {}
        }

    return {
        "messages": [
            AIMessage(
                content="Please reply yes to confirm or no to cancel"
            )
        ]
    }







def researcher_node(state: WebResearchState):
    latest_message = state["messages"][-1]

    # Step 1: Generate targeted search queries
    query_prompt = """Role:
You are a web research agent.

Your task is to generate targeted search queries that will help answer the user's question.

Rules:
- Generate 2 to 3 concise search queries.
- Each query should target a different useful angle of the question.
- Prefer precise technical terms when appropriate.
- Do not answer the user's question.
- Return valid JSON using exactly this structure:
{"queries": ["query 1", "query 2", "query 3"]}
- Do not include explanations or Markdown.
"""

    query_response = llm.invoke([
        SystemMessage(content=query_prompt),
        HumanMessage(content=latest_message.content)
    ])

    try:
        query_decision = json.loads(query_response.content)
        queries = query_decision["queries"]

        if not isinstance(queries, list) or not queries:
            queries = [latest_message.content]

    except (json.JSONDecodeError, KeyError, TypeError):
        queries = [latest_message.content]

    # Step 2: Search the web using all generated queries
    all_results = []

    for query in queries:
        results = search_web(query)
        all_results.extend(results)

    # Step 3: Remove duplicate results
    unique_results = []
    seen_links = set()

    for result in all_results:
        link = result.get("link", "")

        if link and link not in seen_links:
            seen_links.add(link)
            unique_results.append(result)

   

    # Step 4: Build the raw research context
    context = ""

    for result in unique_results:
        context += (
            f"Title: {result.get('title', '')}\n"
            f"Link: {result.get('link', '')}\n"
            f"Snippet: {result.get('snippet', '')}\n"
        )

        sitelinks = result.get("sitelinks", [])

        for sitelink in sitelinks:
            context += (
                f" Sitelink title: {sitelink.get('title', '')}\n"
                f" Sitelink link: {sitelink.get('link', '')}\n"
                f" Sitelink snippet: {sitelink.get('snippet', '')}\n"
            )

        context += "\n"

    # If Serper found nothing useful, stop here
    if not context.strip():
        return {
            "research_context": "",
            "research_error": "No useful web search results were found for that request."
        }

    # Step 5: Let the researcher select the useful evidence
    selection_prompt = """Role:
You are the evidence selector for a web research agent.

Your task is to select the search results that are most useful for answering the user's question.

Rules:
- Select only results that meaningfully help answer the user's question.
- Prefer official, primary, authoritative, or specialist sources when available.
- For niche topics, relevant community sources may be selected when stronger sources are unavailable.
- Do not reject useful evidence solely because it comes from a secondary or community source.
- Remove irrelevant, redundant, or low-information results.
- For comparison questions, retain enough evidence to support each side of the comparison.
- Do not discard a result if it provides a direct definition of one of the concepts being compared.
- Prefer explicit definitions over vague or truncated snippets.
- When multiple sources provide complementary evidence, keep them even if they are about the same topic.
- Do not reduce the evidence set so aggressively that an important concept is supported only by an ambiguous snippet.
- Aim to retain roughly 3 to 6 strong, complementary results when available.

CRITICAL RULES:
- Copy selected evidence exactly from the provided search results.
- Preserve the original title, link, and snippet.
- Do not rewrite, expand, summarize, interpret, complete, or improve snippets.
- Do not infer information that is not explicitly present.
- Do not combine multiple snippets into a new claim.
- Do not answer the user's question.
- Do not write conclusions or explanations.
- Do not add facts from prior knowledge.

Return only selected results using this format:

Title: ...
Link: ...
Snippet: ...

Title: ...
Link: ...
Snippet: ...
"""

    selection_response = llm.invoke([
        SystemMessage(content=selection_prompt),
        HumanMessage(
            content=f"""User question:
{latest_message.content}

Search results:
{context}
"""
        )
    ])

    selected_context = selection_response.content

    if not selected_context.strip():
      return {
         "research_context": "",
         "research_error": "No sufficiently relevant web search was found for that request."
      }

 
    # Step 6: Pass selected evidence to the supervisor
    return {
        "research_context": selected_context,
        "research_error": ""
    }
   

      
       

def report_writer_node(state: WebResearchState):
    latest_message = state["messages"][-1]
    context = state.get("research_context", "")

    

    report_prompt = """Role:
You are the Report Writer for the web research team.

Responsibilities:
Answer the user's question by synthesizing the provided web research results.
Produce the most useful and complete answer that the available evidence supports.
Your job is synthesis, not speculation.

Grounding Rules:
- Use only the provided research results.
- Do not use prior knowledge to fill gaps.
- Do not fabricate facts, calculations, examples, formulas, mechanics, sources, tools, or capabilities.
- Do not infer the meaning of an unfamiliar technical term, statistic, mechanic, or feature from its name.
- If the research does not support a specific claim, do not make that claim.
- If the available evidence is insufficient to establish part of the answer, say so clearly.
- Do not expand a brief source snippet into a more detailed technical explanation unless that explanation is explicitly supported by the provided research.
- Do not convert a broad statement into a more specific mechanic than the evidence supports.

Source Handling:
- Prefer official documentation, primary sources, and authoritative organizations when available.
- When authoritative sources are unavailable, relevant specialist websites, community research, forums, and other secondary sources may be used.
- Clearly qualify information that is uncertain, disputed, or based primarily on community research.
- Do not attribute a claim to a source unless that source actually supports it.

Technical Information:
- Preserve important terminology, definitions, formulas, calculations, measurements, and distinctions found in the research.
- Do not replace specific technical information with generic interpretations or advice.
- For technical or niche topics, prioritize concrete documented or measured behavior over speculation.
- Do not invent recommendations or conclusions that are not supported by the research.
- When evidence is limited to short search snippets, preserve the level of certainty and detail of those snippets instead of elaborating beyond them.
- Do not infer the exact order of operations in a formula unless the research explicitly states it.
- If the research establishes that one value is additive and another is multiplicative, state only that distinction unless the calculation order is directly supported.
- Do not infer how modifiers interact with other modifiers unless the research explicitly describes that interaction.

Current Information:
When the user asks for the latest, newest, or current information:
- Prefer official or primary sources when available.
- For software versions, clearly distinguish stable releases from beta, preview, or prerelease versions.
- Do not rely on prior knowledge for current information.

Response Style:
- Answer the user's actual question directly.
- Be clear, accurate, and appropriately detailed.
- Mention relevant sources when useful.
- Never mention other agents, specialists, assistants, teams, or handoffs in the final answer.
- Never suggest transferring, redirecting, or handing the user to another agent or specialist.
- Do not end the response by offering additional services, further research, specialist help, or follow-up assistance.
- End the response naturally after answering the user's question.
"""

    

    response = llm.invoke([
        SystemMessage(content=report_prompt),
        HumanMessage(
            content=f"""
Research results:
{context}

User question:
{latest_message.content}
"""
        )
    ])

    return {
        "messages": [response]
    }


def web_research_supervisor_node(state: WebResearchState):

   research_context = state.get("research_context", "")
   research_error = state.get("research_error", "")

   if not research_context and not research_error:
      return {
         "web_next_step": "researcher"
      }

   if research_error:
      return {
         "web_next_step": "fallback"
      }

   latest_message = state ["messages"][-1]

   supervisor_prompt = """Role:
You are the Web Research Supervisor.

Responsibilities:
Evaluate the collected research and decide whether it provides enough relevant evidence to answer the user's question.

Possible Decisions:
- report_writer
- fallback

Decision Rules:
- Choose "report_writer" when the research contains enough relevant evidence to provide a useful answer, even if the answer must be qualified or incomplete.
- A direct definition, comparison, relationship, measured behavior, or supported distinction between the concepts in the user's question is sufficient for "report_writer".
- Do not require complete formulas, exhaustive documentation, or authoritative confirmation when the available evidence still supports a meaningful answer.
- Choose "fallback" when the research is empty, unrelated, clearly unreliable, contradictory without enough evidence to resolve the question, or genuinely insufficient to provide a useful answer.
- Do not require every detail of the user's question to be answered perfectly. Choose "report_writer" when a useful, appropriately qualified answer can be produced from the available evidence.
- Source quality alone is not a reason to reject otherwise relevant research.
- Prefer official, primary, and authoritative sources when available.
- When authoritative sources are unavailable, relevant specialist websites, community research, forums, and other secondary sources may still provide sufficient evidence.
- Do not require official documentation for every topic.
- Do not answer the user's question yourself. Your only task is to select the next step.

Output:
Return valid JSON using exactly this structure:
{"next_step": "report_writer"}

The value of "next_step" must be either "report_writer" or "fallback".
Do not include explanations, Markdown, or any other text.
"""

   response = llm.invoke([
      SystemMessage(content=supervisor_prompt),
      HumanMessage(
         content=f"""
User question:
{latest_message.content}

Research results:
{research_context}
"""
      )
   ])

   try:
      decision = json.loads(response.content)

      next_step = decision["next_step"].strip().lower()

      if next_step not in {"report_writer", "fallback"}:
         next_step = "fallback"

   except (json.JSONDecodeError, KeyError, AttributeError):
      next_step = "fallback"

   return {
      "web_next_step": next_step
   }         












   
def visualization_node(state: GraphState):
   latest_message = state["messages"][-1]
   
   visualization_prompt = f"""Role:
   You are the Visualization agent.
   Extract the chart type, labels, and numeric values needed to create a chart from the user's request.
   
   Supported chart types:
   - bar
   - pie
   - line
   
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

      request_text = latest_message.content.lower()

      if "pie" in request_text:
         chart_type = "pie"
      elif "line" in request_text:
         chart_type = "line"
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
      ],
      "agent_results": {
         **state.get("agent_results", {}),
         "chart": {
            "chart_type": chart_type,
            "labels": labels,
            "values": values
         }
      }
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