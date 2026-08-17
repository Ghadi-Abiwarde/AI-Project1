from graph import graph
from langchain_core.messages import HumanMessage


def main():
    config = {
        "configurable": {
            "thread_id": "user-1"
        }
    }

    print("AI Assistant")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("YOU: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config
        )

        print("ASSISTANT:", result["messages"][-1].content)
        print()


if __name__ == "__main__":
    main()