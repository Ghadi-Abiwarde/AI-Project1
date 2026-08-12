from graph import graph
from langchain_core.messages import HumanMessage
from state import GraphState
config = {
    "configurable": {
        "thread_id": "test-user-1"
    }
}

#test_prompts = [
 #   "What does the employee handbook say about annual leave?",
  #  "How many remote work days are employees allowed per week?",
   # "Can employees work remotely from another country?",
    #

prompt = "Create a bar chart showing total spending by customer."

result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=prompt)
            ]
        },
        config=config
    )
print(result["next_agent"])
print("ASSISTANT:", result["messages"][-1].content)