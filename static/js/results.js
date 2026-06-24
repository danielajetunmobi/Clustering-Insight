(function () {
    const dataElement = document.getElementById("analysis-data");
    if (!dataElement) {
        return;
    }

    const chartData = JSON.parse(dataElement.textContent);

    function baseLayout() {
        return {
            margin: { t: 24, r: 20, b: 48, l: 56 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "#ffffff",
            font: { family: "Inter, Segoe UI, Arial, sans-serif", color: "#17212b" },
            xaxis: { gridcolor: "#e9edf1", zerolinecolor: "#d8dee4" },
            yaxis: { gridcolor: "#e9edf1", zerolinecolor: "#d8dee4" }
        };
    }

    function plotConfig() {
        return { responsive: true, displaylogo: false };
    }

    function renderScatter(points) {
        const target = document.getElementById("scatterPlot");
        if (!target) {
            return;
        }
        if (!window.Plotly) {
            target.innerHTML = "<div class='chart-fallback'>Plotly.js is unavailable.</div>";
            return;
        }

        const clusters = Array.from(new Set(points.map((point) => point.cluster))).sort();
        const traces = clusters.map((cluster) => {
            const rows = points.filter((point) => point.cluster === cluster && point.anomaly !== "anomaly");
            return {
                x: rows.map((point) => point.x),
                y: rows.map((point) => point.y),
                text: rows.map((point) => `Row ${point.row}`),
                mode: "markers",
                type: "scatter",
                name: `Cluster ${cluster}`,
                marker: { size: 10, line: { color: "#ffffff", width: 1 } },
                hovertemplate: "%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra>%{fullData.name}</extra>"
            };
        });

        const anomalyRows = points.filter((point) => point.anomaly === "anomaly");
        if (anomalyRows.length) {
            traces.push({
                x: anomalyRows.map((point) => point.x),
                y: anomalyRows.map((point) => point.y),
                text: anomalyRows.map((point) => `Row ${point.row}`),
                mode: "markers",
                type: "scatter",
                name: "Anomaly",
                marker: { color: "#c2410c", size: 13, symbol: "x", line: { width: 2 } },
                hovertemplate: "%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra>Anomaly</extra>"
            });
        }

        const layout = baseLayout();
        layout.xaxis.title = "Component X";
        layout.yaxis.title = "Component Y";
        layout.legend = { orientation: "h", y: -0.18 };
        Plotly.newPlot(target, traces, layout, plotConfig());
    }

    function renderDistribution(distribution) {
        const target = document.getElementById("distributionChart");
        if (!target || !window.Plotly) {
            return;
        }
        const layout = baseLayout();
        layout.xaxis.title = "Cluster";
        layout.yaxis.title = "Rows";
        Plotly.newPlot(
            target,
            [{
                x: distribution.map((item) => item.label),
                y: distribution.map((item) => item.count),
                type: "bar",
                marker: { color: "#0f766e" },
                hovertemplate: "Cluster %{x}<br>Rows=%{y}<extra></extra>"
            }],
            layout,
            plotConfig()
        );
    }

    function renderElbow(elbow) {
        const target = document.getElementById("elbowChart");
        if (!target) {
            return;
        }
        if (!elbow.length) {
            target.innerHTML = "<div class='chart-fallback'>Elbow data is available for K-Means runs.</div>";
            return;
        }
        if (!window.Plotly) {
            return;
        }
        const layout = baseLayout();
        layout.xaxis.title = "K";
        layout.yaxis.title = "Inertia";
        Plotly.newPlot(
            target,
            [{
                x: elbow.map((item) => item.k),
                y: elbow.map((item) => item.inertia),
                type: "scatter",
                mode: "lines+markers",
                line: { color: "#2563eb", width: 3 },
                marker: { size: 8 },
                hovertemplate: "K=%{x}<br>Inertia=%{y:.3f}<extra></extra>"
            }],
            layout,
            plotConfig()
        );
    }

    renderScatter(chartData.points || []);
    renderDistribution(chartData.distribution || []);
    renderElbow(chartData.elbow || []);
})();
