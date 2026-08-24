import { useEffect, useRef } from "react";

export default function InteractiveChart() {
    const canvasRef = useRef(null);

    useEffect(() => {
        let chartInstance;

        const loadChart = async () => {
            if (!window.Chart) {
                await new Promise((resolve, reject) => {
                    const script = document.createElement("script");
                    script.src =
                        "https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js";

                    script.onload = resolve;
                    script.onerror = reject;

                    document.head.appendChild(script);
                });
            }

            const ctx = canvasRef.current;

            chartInstance = new window.Chart(ctx, {
                type: props.chart_type,

                data: {
                    labels: props.labels,
                    datasets: [
                        {
                            label: "Values",
                            data: props.values
                        }
                    ]
                },

                options: {
                    responsive: true,
                    maintainAspectRatio: false,

                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    const rawValue = context.raw;

                                    if (props.chart_type === "pie") {
                                        const total = props.values.reduce(
                                            (sum, value) => sum + value,
                                            0
                                        );

                                        if (total === 0){
                                            return `${context.label}: ${rawValue}`;
                                        }

                                        const percentage = (
                                            (rawValue / total) *100
                                        ).toFixed(1);

                                        return `${context.label}: ${rawValue} (${percentage}%)`;

                                        
                                    }

                                    return `${context.label}: ${rawValue}`;
                                }
                            }
                        }
                    }
                }
            });
        };

        loadChart();

        return () => {
            if (chartInstance) {
                chartInstance.destroy();
            }
        };
    }, [props.chart_type, props.labels, props.values]);

    return (
        <div style={{ height: "400px", width: "100%" }}>
            <canvas ref={canvasRef}></canvas>
        </div>
    );
}