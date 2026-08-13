from visualization import create_chart
from nodes import visualization_node, supervisor_node, rag_node

from langchain_core.messages import HumanMessage
test_state = {
    "messages": [
        HumanMessage(
            content="How many annual leave days do employees receive?"
        )
    ]
}

result = rag_node(test_state)

print(result["messages"][-1].content)