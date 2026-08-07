from graph import graph
from langchain_core.messages import HumanMessage

config = {
    "configurable": {
        "thread_id": "test-user-1"
    }
}

test_prompts = [
    "What does the employee handbook say about annual leave?",
    "How many remote work days are employees allowed per week?",
    "Can employees work remotely from another country?",
    "What health insurance company does Nexora use?"
]

for prompt in test_prompts:
    print("\nUSER:", prompt)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=prompt)
            ]
        },
        config=config
    )

    print("ASSISTANT:", result["messages"][-1].content)