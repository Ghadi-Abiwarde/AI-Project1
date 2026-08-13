from langchain_core.messages import HumanMessage
from nodes import conversation_node
from nodes import sql_node, web_research_node
from database import execute_query

test_state = {
    "messages": [
        HumanMessage(content="Which customer placed the most orders?")
    ]
}
try:
    result = execute_query(
        "SELECT definitely_not_a_column FROM customers;"
    )
    print("Database results:", result["messages"][-1].content)
except Exception as error:
    print("SQL execution error:", error)
#result = sql_node(test_state)
 

#print(result["messages"][0].content)