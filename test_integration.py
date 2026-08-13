from graph import graph
from langchain_core.messages import HumanMessage

test_cases = [
   {
    "prompt": "Which customer spent the most money?",
    "expected_route": "sql"
},
{
    "prompt": "Show me a bar chart of the current stock of each product.",
    "expected_route": "sql",
    "expected_visualization": True
}
]
for index, test in enumerate(test_cases):

    prompt = test["prompt"]
    expected_route = test["expected_route"]

    config = {
        "configurable": {
            "thread_id": f"integration-test-{index}"
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=prompt)
            ]
        },
        config=config
    )

    actual_route = result["next_agent"]

    print("\nPrompt:", prompt)
    print("Expected route:", expected_route)
    print("Actual route:", actual_route)

    if actual_route == expected_route:
        print("PASS")
    else:
        print("FAIL")

    if "expected_visualization" in test:
        print(
            "Expected visualization:",
            test["expected_visualization"]
        )
        print(
            "Actual visualization:",
            result.get("needs_visualization")
        )