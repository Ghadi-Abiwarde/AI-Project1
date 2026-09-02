import matplotlib.pyplot as plt
from decimal import Decimal


def create_chart(chart_type, labels, values):
    fig, ax = plt.subplots(figsize=(8, 5))

    def format_pie_label(percentage):
        total = sum(values)
        raw_value = percentage * total / 100

        return f"{percentage:.1f}%\n({raw_value:g})"

    if chart_type == "bar":
        ax.bar(labels, values)

    elif chart_type == "pie":
        ax.pie(
            values,
            labels=labels,
            autopct=format_pie_label
        )

    elif chart_type == "line":
        ax.plot(
            labels,
            values,
            marker="o"
        )    

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig.tight_layout()

    return fig    

def validate_chart_data(chart_type, labels, values):

    if chart_type not in {"bar", "pie", "line"}:
        return "Unsupported chart type."

    if not labels or not values:
        return "No data is available to create the chart."

    if len(labels) != len(values):
        return "The chart labels and values do not match."

    if not all(isinstance(value, (int, float, Decimal)) for value in values):
        return "Chart values must be numeric."

    if chart_type == "pie" and sum(values) == 0:
        return "A pie chart cannot be created when all values are zero."

    return None

