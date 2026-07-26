import { useMemo, useState } from "react";
import "../../style/forecast-page.css";
import { ChartAreaStep, type ChartDataRow } from "./chart";
import { ForecastConfiguration, type ForecastSettings } from "./components/ForecastConfiguration";
import { ForecastTable } from "./components/ForecastTable";
import { ForecastExportActions, ForecastInsights, ForecastSummary, InventoryImpact, ModelPerformance } from "./components/ForecastPanels";
import { FORECAST_PRODUCTS, formatDate, generateForecastData, type ForecastPoint } from "./forecastData";

const defaultSettings: ForecastSettings = {
  productId: "PRD-001",
  warehouse: "Central Fresh Distribution",
  store: "All stores",
  horizon: 30,
  startDate: "2026-07-27",
  model: "Best available model",
  promotion: true,
  discount: 10,
  price: 31_500,
};

const weeklySummary = (data: ForecastPoint[]): ChartDataRow[] => {
  const result: ChartDataRow[] = [];
  for (let index = 0; index < data.length; index += 7) {
    const week = data.slice(index, index + 7);
    result.push({ month: `${formatDate(week[0].date, true)}–${formatDate(week.at(-1)!.date, true)}`, data: week.reduce((sum, point) => sum + point.predictedUnitsSold, 0) });
  }
  return result;
};

const DemandForecast = () => {
  const [settings, setSettings] = useState(defaultSettings);
  const [forecast, setForecast] = useState(() => generateForecastData(defaultSettings.horizon, defaultSettings.startDate, defaultSettings.promotion, defaultSettings.discount, defaultSettings.price));
  const [runState, setRunState] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [view, setView] = useState<"daily" | "weekly">("daily");
  const product = FORECAST_PRODUCTS.find(item => item.id === settings.productId) ?? FORECAST_PRODUCTS[0];
  const chartData = useMemo<ChartDataRow[]>(() => view === "daily"
    ? forecast.map(point => ({ month: formatDate(point.date, true), data: point.predictedUnitsSold }))
    : weeklySummary(forecast), [forecast, view]);

  const generate = () => {
    setRunState("loading");
    window.setTimeout(() => {
      if (!settings.startDate || settings.price <= 0) {
        setRunState("error");
        return;
      }
      try {
        setForecast(generateForecastData(settings.horizon, settings.startDate, settings.promotion, settings.discount, settings.price));
        setRunState("success");
        setView("daily");
      } catch {
        setRunState("error");
      }
    }, 650);
  };
  const reset = () => { setSettings(defaultSettings); setForecast(generateForecastData(defaultSettings.horizon, defaultSettings.startDate, defaultSettings.promotion, defaultSettings.discount, defaultSettings.price)); setRunState("idle"); setView("daily"); };

  return <main className="forecast-page">
    <header className="forecast-hero">
      <div className="forecast-shell forecast-hero__layout">
        <div><p className="forecast-kicker">Planning workspace / Demand intelligence</p><h1>Demand Forecast</h1><p>Predict customer demand and plan inventory requirements for upcoming periods.</p></div>
        <dl className="dataset-status"><div><dt>Dataset</dt><dd>Perishable Goods Sales</dd></div><div><dt>Last updated</dt><dd>July 25, 2026</dd></div><div><dt>Status</dt><dd><span className="forecast-badge forecast-badge--success">Ready for forecasting</span></dd></div></dl>
      </div>
    </header>
    <div className="forecast-shell forecast-content">
      <ForecastConfiguration settings={settings} onChange={setSettings} onGenerate={generate} onReset={reset} state={runState} />
      <ForecastSummary data={forecast} />
      <section className="forecast-section forecast-chart-section" aria-labelledby="chart-heading">
        <div className="forecast-chart-header"><div><span className="forecast-index">03 · Forecast result</span><h2 id="chart-heading">{product.name}</h2><p>{settings.warehouse} · Southeast region</p><div className="chart-context"><span>{formatDate(forecast[0].date)} – {formatDate(forecast.at(-1)!.date)}</span><span className="forecast-badge forecast-badge--dark">{settings.horizon}-day horizon</span></div></div><div className="chart-view-control" role="group" aria-label="Chart aggregation"><button type="button" className={view === "daily" ? "active" : ""} onClick={() => setView("daily")}>Daily</button><button type="button" className={view === "weekly" ? "active" : ""} onClick={() => setView("weekly")}>Weekly summary</button></div></div>
        <div className="chart-legend"><span><i className="legend-line" />Predicted units sold</span>{view === "daily" && <small>Hover or focus any day for its exact date and value. Scroll horizontally for longer horizons.</small>}{view === "weekly" && <small>Weekly totals are calculated separately; the daily forecast remains unchanged.</small>}</div>
        <div className="forecast-chart-card"><ChartAreaStep chartData={chartData} /></div>
      </section>
      <ForecastInsights data={forecast} promoted={settings.promotion} />
      <ForecastTable data={forecast} />
      <InventoryImpact data={forecast} />
      <ModelPerformance model={settings.model} />
      <ForecastExportActions data={forecast} />
    </div>
  </main>;
};

export { DemandForecast };
