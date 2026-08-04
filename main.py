from langchain_core.messages import HumanMessage
from nodes import conversation_node

test_state = {
    "messages": [
        HumanMessage(content="Explain what a supervisor agent does.")
    ]
}

result = conversation_node(test_state)

print(result)
print(result["messages"][0].content)