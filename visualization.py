import matplotlib.pyplot as plt

def create_chart(chart_type, labels, values):
    plt.figure(figsize=(8 , 5))
    if chart_type == "bar":
        plt.bar(labels, values)
    elif chart_type == "pie":
        plt.pie(values, labels = labels, autopct="%1.1f%%")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    plt.tight_layout()
    plt.show()        

def validate_chart_data(chart_type, labels, values):

    if chart_type not in {"bar", "pie"}:
        return "Unsupported chart type."

    if len(labels) != len(values):
        return "The chart labels and values do not match."

    if not all(isinstance(value, (int, float)) for value in values):
        return "Chart values must be numeric."

    return None

