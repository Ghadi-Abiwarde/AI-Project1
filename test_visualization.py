from visualization import create_chart
from nodes import visualization_node, supervisor_node

from langchain_core.messages import HumanMessage
test_state = {
    "messages": [
        HumanMessage(
            content="Which customer spent the most money"
        )
    ]
}

result = supervisor_node(test_state)

print(result)