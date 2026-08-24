import chainlit as cl
import matplotlib.pyplot as plt

from graph import graph
from visualization import create_chart
from langchain_core.messages import HumanMessage




@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Database question",
            message="Which customer spent the most money?"
        ),
        cl.Starter(
            label="Company policy",
            message="How many annual leave days do employees receive?"
        ),
        cl.Starter(
            label="Web research",
            message="What is the latest stable PostgreSQL version?"
        ),
        cl.Starter(
            label="Create a chart",
            message="Create a bar chart showing total spending by customer."
        ),
    ]


@cl.on_message
async def main(message: cl.Message):

    config = {
        "configurable": {
            "thread_id": cl.context.session.id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=message.content)
            ],
            "agent_results": {},
            "needs_visualization": False
        },
        config=config
    )

    response = result["messages"][-1].content

    chart = result.get("agent_results", {}).get("chart")

    if chart:
        element = cl.CustomElement(
            name="InteractiveChart",
            props={
                "chart_type": chart["chart_type"],
                "labels": chart["labels"],
                "values": chart["values"]
            },
            display="inline"
        )

        await cl.Message(
            content=response,
            elements=[element]
        ).send()

    else:
        await cl.Message(
            content=response
        ).send()