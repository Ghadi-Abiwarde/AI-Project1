from langchain_core.messages import HumanMessage
from nodes import supervisor_node


test_prompts = [
    ("Hello, how are you?","conversation"),
    ("Which customer spent the most money?","sql"),
    ("Research online the latest developments in artificial intelligence.", "web_research"),
    ("Create a pie chart from A=40, B=35, C=25.", "visualization"),
    ("What does the employee handbook say about annual leave?", "rag"),
    ("Research current annual leave laws in Lebanon.", "web_research")
]
passed = 0
for prompt, expected_route in test_prompts:
 test_state = {
    "messages": [
        HumanMessage(content=prompt)
    ]
}
 result = supervisor_node(test_state)
 actual_route = result["next_agent"]
 if actual_route == expected_route:
    print("PASS")
    passed +=1
 else:
    print(f"FAIL | {prompt}")
    print(f"Expected: {expected_route}")
    print(f"Actual: {actual_route}")

print(f"\nPassed: {passed}/{len(test_prompts)}")
print(result["next_agent"])
 