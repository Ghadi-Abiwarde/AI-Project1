from graph import graph
from langchain_core.messages import HumanMessage
from state import GraphState
config = {
    "configurable": {
        "thread_id": "test-user-1"
    }
}

test_prompts = [
    "how many annual leave days do employees receive?",
    #"What is the company's remote work policy?",
    #"Can I work remotely from another country?",
    "What happens if an employee is repeatedly late?",
    #"What health insurance provider does the company use?"
    #"What changed in the latest Python release?",
    #"What are the latest developments in artificial intelligence?",
    "What is the current stable version of PostgreSQL?",
    "Create a pie chart from A=40, B=35, C=25.",
    "Show me a bar chart of the current stock of each product.",
    "Which customer spent the most money?",
    #"According to company policy, can employees work remotely from another country?",
    #"What is the latest stable PostgreSQL version?",
    "Create a bar chart showing total spending by customer."
]

#prompt = "Create a bar chart showing total spending by customer."
for prompt in test_prompts:
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=prompt)
            ]
        },
        config=config
    )
    print("ASSISTANT:", result["messages"][-1].content)
    #print("ROUTE:", result["next_agent"])
    #print("NEEDS VISUALIZATION:", result["needs_visualization"])