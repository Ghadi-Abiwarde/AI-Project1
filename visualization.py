import matplotlib.pyplot as plt
from decimal import Decimal

def create_chart(chart_type, labels, values):
    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "bar":
        ax.bar(labels, values)

    elif chart_type == "pie":
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%"
        )

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig.tight_layout()

    return fig    

def validate_chart_data(chart_type, labels, values):

    if chart_type not in {"bar", "pie"}:
        return "Unsupported chart type."

    if not labels or not values:
        return "No data is available to create the chart."

    if len(labels) != len(values):
        return "The chart labels and values do not match."

    if not all(isinstance(value, (int, float, Decimal)) for value in values):
        return "Chart values must be numeric."

    return None

