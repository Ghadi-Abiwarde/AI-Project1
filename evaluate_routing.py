from graph import graph
from langchain_core.messages import HumanMessage
from langsmith import Client

client = Client()

def target(inputs: dict) -> dict:
    prompt = inputs["prompt"]

    config = {
        "configurable": {
            "thread_id": f"evaluation-{prompt}"
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

    return {
        "route": result["next_agent"],
        "visualization": result.get("needs_visualization", False)
    }

def routing_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> bool:

    route_correct = (
        outputs["route"]
        == reference_outputs["expected_route"]
    )

    visualization_correct = (
        outputs["visualization"]
        == reference_outputs["expected_visualization"]
    )

    return route_correct and visualization_correct

results = client.evaluate(
    target,
    data="multi-agent-routing-evaluation",
    evaluators=[routing_evaluator],
    experiment_prefix="routing-eval"
)

print(results)