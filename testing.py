import json

def test_supervisor_parsing(fake_response_content):
    try:
        decision = json.loads(fake_response_content)

        route = decision["next_agent"].strip().lower()
        needs_visualization = decision["needs_visualization"]

    except (json.JSONDecodeError, KeyError, AttributeError):
        route = "conversation"
        needs_visualization = False

    allowed_routes = {
        "conversation",
        "sql",
        "web_research",
        "visualization",
        "rag",
    }

    if route not in allowed_routes:
        route = "conversation"
        needs_visualization = False

    if not isinstance(needs_visualization, bool):
        needs_visualization = False

    return {
        "next_agent": route,
        "needs_visualization": needs_visualization
    }

print(test_supervisor_parsing(
    '{"next_agent": "banana"}'
))