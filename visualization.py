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
   

