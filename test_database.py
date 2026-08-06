from langchain_core.messages import HumanMessage
from nodes import conversation_node
from nodes import sql_node
from database import execute_query

test_state = {
    "messages": [
        HumanMessage(content="How many products are in the Accessories category?")
    ]
}

result = sql_node(test_state)

print("Database results:", result)

#print(result["messages"][0].content)