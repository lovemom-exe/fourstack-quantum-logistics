import { calculateInventoryImpact, formatDate, summarizeForecast, toForecastCsv, type ForecastPoint } from "../forecastData";

export const ForecastSummary = ({ data }: { data: ForecastPoint[] }) => {
  const summary = summarizeForecast(data);
  const cards = [
    ["Forecast horizon", `${data.length} days`, `${formatDate(data[0].date, true)} – ${formatDate(data.at(-1)!.date, true)}`],
    ["Total predicted demand", `${summary.total.toLocaleString()} units`, "Across the selected period"],
    ["Average daily demand", `${summary.average.toLocaleString()} units`, "Predicted units per day"],
    ["Peak demand", `${summary.peak.predictedUnitsSold.toLocaleString()} units`, formatDate(summary.peak.date)],
    ["Lowest demand", `${summary.lowest.predictedUnitsSold.toLocaleString()} units`, formatDate(summary.lowest.date)],
    ["Forecast trend", `${summary.trendPct >= 0 ? "+" : ""}${summary.trendPct}%`, summary.trendPct >= 0 ? "Increasing" : "Decreasing"],
    ["Confidence level", `${summary.confidence}%`, "Demonstration interval"],
  ];
  return <section className="forecast-section" aria-labelledby="summary-heading"><div className="forecast-section__heading"><div><span className="forecast-index">02</span><h2 id="summary-heading">Forecast summary</h2><p>Headline demand signals for the selected period.</p></div><span className="forecast-badge forecast-badge--success">Forecast complete</span></div><div className="forecast-summary-grid">{cards.map(([label, value, note]) => <article key={label}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>)}</div></section>;
};

export const ForecastInsights = ({ data, promoted }: { data: ForecastPoint[]; promoted: boolean }) => {
  const summary = summarizeForecast(data);
  const impact = calculateInventoryImpact(data);
  const peakStart = data[Math.max(0, data.indexOf(summary.peak) - 3)].date;
  const insights = [
    `Demand is expected to ${summary.trendPct >= 0 ? "increase" : "decrease"} by approximately ${Math.abs(summary.trendPct)}% across the forecast period.`,
    `The strongest demand window is expected between ${formatDate(peakStart, true)} and ${formatDate(summary.peak.date, true)}.`,
    promoted ? "The planned promotion overlaps with elevated demand and may require earlier replenishment." : "No planned promotion has been included in this demonstration forecast.",
    impact.stockoutDate ? `Current and incoming inventory may be insufficient from ${formatDate(impact.stockoutDate)}.` : "Current and incoming inventory cover the predicted demand in this period.",
  ];
  return <section className="forecast-insights" aria-labelledby="insights-heading"><div><span className="forecast-index">04 · Mock interpretation</span><h2 id="insights-heading">Forecast Insights</h2><p>Operational guidance generated from demonstration forecast rules—not conclusions from a live model.</p></div><ul>{insights.map((insight, index) => <li key={insight}><span>{String(index + 1).padStart(2, "0")}</span>{insight}</li>)}</ul></section>;
};

export const InventoryImpact = ({ data }: { data: ForecastPoint[] }) => {
  const impact = calculateInventoryImpact(data);
  const state = impact.shortage > 0 ? "Stockout expected" : impact.remaining < impact.demand * .15 ? "Low stock risk" : "Sufficient stock";
  return <section className="forecast-section inventory-impact" aria-labelledby="inventory-impact-heading"><div className="forecast-section__heading"><div><span className="forecast-index">06</span><h2 id="inventory-impact-heading">Inventory impact</h2><p>Estimated stock position based on mock forecast demand.</p></div><span className={`forecast-badge ${state === "Sufficient stock" ? "forecast-badge--success" : state === "Low stock risk" ? "forecast-badge--warning" : "forecast-badge--error"}`}>{state}</span></div><div className="inventory-flow"><div><span>Current inventory</span><strong>{impact.current.toLocaleString()}</strong><small>units</small></div><i>+</i><div><span>Incoming inventory</span><strong>{impact.incoming.toLocaleString()}</strong><small>units</small></div><i>−</i><div><span>Predicted demand</span><strong>{impact.demand.toLocaleString()}</strong><small>units</small></div><i>=</i><div className={impact.remaining < 0 ? "inventory-negative" : ""}><span>Estimated remaining</span><strong>{impact.remaining.toLocaleString()}</strong><small>units</small></div></div><dl className="inventory-actions"><div><dt>Expected stockout</dt><dd>{impact.stockoutDate ? formatDate(impact.stockoutDate) : "Not expected"}</dd></div><div><dt>Estimated shortage</dt><dd>{impact.shortage.toLocaleString()} units</dd></div><div><dt>Suggested reorder estimate</dt><dd>{impact.reorder.toLocaleString()} units</dd></div></dl><p className="estimate-note">Reorder quantity is a planning estimate until backend inventory optimization is connected.</p></section>;
};

export const ModelPerformance = ({ model }: { model: string }) => <section className="forecast-section" aria-labelledby="performance-heading"><div className="forecast-section__heading"><div><span className="forecast-index">07</span><h2 id="performance-heading">Model performance</h2><p>Demonstration metadata—not measured results from a live forecast run.</p></div><span className="forecast-badge forecast-badge--neutral">Mock metrics</span></div><div className="model-metadata"><article><span>Model name</span><strong>{model === "Best available model" ? "XGBoost · selected" : model}</strong></article><article><span>Features</span><strong>42</strong></article><article><span>Training date</span><strong>July 20, 2026</strong></article><article><span>Training data</span><strong>Jan 2024 – Jun 2026</strong></article><article className="metric-card"><span>R²</span><strong>0.87</strong><small>Demand variation explained</small></article><article className="metric-card"><span>MAE</span><strong>31 units</strong><small>Average absolute error</small></article><article className="metric-card"><span>RMSE</span><strong>44 units</strong><small>Weights larger errors more</small></article></div><div className="metric-explanation"><p><strong>R²</strong> shows how much demand variation the model explains.</p><p><strong>MAE</strong> shows the average prediction error in units.</p><p><strong>RMSE</strong> gives more weight to larger errors.</p></div></section>;

const download = (name: string, content: string, type: string) => {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url);
};
export const ForecastExportActions = ({ data }: { data: ForecastPoint[] }) => {
  const summary = summarizeForecast(data);
  const summaryText = `Demand forecast: ${data.length} days, ${summary.total.toLocaleString()} total predicted units, ${summary.average.toLocaleString()} average units/day, peak ${summary.peak.predictedUnitsSold.toLocaleString()} units on ${formatDate(summary.peak.date)}. Demonstration data.`;
  return <section className="export-panel" aria-labelledby="export-heading"><div><span className="forecast-index">08</span><h2 id="export-heading">Export forecast</h2><p>Share or archive the currently displayed demonstration results.</p></div><div><button type="button" className="forecast-button forecast-button--primary" onClick={() => download("demand-forecast.csv", toForecastCsv(data), "text/csv")}>Download forecast CSV</button><button type="button" className="forecast-button" onClick={() => download("forecast-report.txt", `${summaryText}\n\nGenerated by the frontend demonstration.`, "text/plain")}>Download forecast report</button><button type="button" className="forecast-button" onClick={() => navigator.clipboard.writeText(summaryText)}>Copy summary</button><button type="button" className="forecast-button" onClick={() => window.print()}>Print view</button></div></section>;
};
