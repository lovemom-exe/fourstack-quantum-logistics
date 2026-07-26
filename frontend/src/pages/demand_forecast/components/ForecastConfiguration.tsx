import { useState, type FormEvent } from "react";
import { FORECAST_PRODUCTS, type Horizon } from "../forecastData";

export interface ForecastSettings {
  productId: string;
  warehouse: string;
  store: string;
  horizon: Horizon;
  startDate: string;
  model: string;
  promotion: boolean;
  discount: number;
  price: number;
}

const modelDescriptions: Record<string, string> = {
  "Best available model": "Automatically selects the strongest validated model for this product.",
  "Random Forest": "A robust classical ensemble for non-linear demand patterns.",
  XGBoost: "Gradient-boosted trees tuned for promotions and seasonal effects.",
  VQR: "Quantile regression focused on uncertainty ranges and demand risk.",
};
const plans: { days: Horizon; plan: string }[] = [{ days: 14, plan: "Starter" }, { days: 30, plan: "Professional" }, { days: 60, plan: "Business" }, { days: 90, plan: "Enterprise" }];

export const ForecastConfiguration = ({ settings, onChange, onGenerate, onReset, state }: { settings: ForecastSettings; onChange: (settings: ForecastSettings) => void; onGenerate: () => void; onReset: () => void; state: "idle" | "loading" | "success" | "error" }) => {
  const [query, setQuery] = useState("");
  const selected = FORECAST_PRODUCTS.find((product) => product.id === settings.productId) ?? FORECAST_PRODUCTS[0];
  const visibleProducts = FORECAST_PRODUCTS.filter((product) => `${product.name} ${product.category}`.toLowerCase().includes(query.toLowerCase()));
  const update = <K extends keyof ForecastSettings>(key: K, value: ForecastSettings[K]) => onChange({ ...settings, [key]: value });
  const submit = (event: FormEvent) => { event.preventDefault(); onGenerate(); };

  return <section className="forecast-section" aria-labelledby="configuration-heading">
    <div className="forecast-section__heading"><div><span className="forecast-index">01</span><h2 id="configuration-heading">Forecast configuration</h2><p>Choose the operating scope and optional future conditions.</p></div><span className="forecast-badge forecast-badge--neutral">Local demonstration</span></div>
    <form className="forecast-form" onSubmit={submit}>
      <div className="forecast-form__group forecast-form__group--product">
        <div className="group-title"><strong>Product selection</strong><span>Required</span></div>
        <label><span>Search products</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search name or category" /></label>
        <label><span>Product</span><select value={settings.productId} onChange={(event) => update("productId", event.target.value)}>{visibleProducts.map((product) => <option key={product.id} value={product.id}>{product.name} — {product.category}</option>)}</select></label>
        <div className="selected-product"><div className="product-monogram">{selected.name.slice(0, 2).toUpperCase()}</div><div><strong>{selected.name}</strong><span>{selected.category}</span></div><div><strong>{selected.stock.toLocaleString()}</strong><span>units currently available</span></div></div>
      </div>
      <div className="forecast-form__group">
        <div className="group-title"><strong>Location</strong><span>Required</span></div>
        <label><span>Warehouse</span><select value={settings.warehouse} onChange={(event) => update("warehouse", event.target.value)}><option>Central Fresh Distribution</option><option>North Cold Storage</option><option>Mekong Regional Hub</option></select></label>
        <label><span>Store</span><select value={settings.store} onChange={(event) => update("store", event.target.value)}><option>All stores</option><option>District 1 Flagship</option><option>Thu Duc Store</option><option>District 7 Store</option></select></label>
        <div className="location-note"><span>Region</span><strong>Southeast · Ho Chi Minh City</strong></div>
      </div>
      <div className="forecast-form__group forecast-form__group--wide">
        <div className="group-title"><strong>Forecast horizon</strong><span>Plan-based options</span></div>
        <div className="horizon-picker">{plans.map(({ days, plan }) => <label key={days}><input type="radio" name="horizon" checked={settings.horizon === days} onChange={() => update("horizon", days)} /><span><strong>{days} days</strong><small>{plan}</small></span></label>)}</div>
      </div>
      <div className="forecast-form__group">
        <div className="group-title"><strong>Timing & model</strong><span>Required</span></div>
        <label><span>Forecast start date</span><input type="date" required value={settings.startDate} onChange={(event) => update("startDate", event.target.value)} /></label>
        <label><span>Model selection</span><select value={settings.model} onChange={(event) => update("model", event.target.value)}>{Object.keys(modelDescriptions).map(model => <option key={model}>{model}</option>)}</select></label>
        <p className="model-description">{modelDescriptions[settings.model]}</p>
      </div>
      <fieldset className="forecast-form__group future-conditions">
        <legend className="group-title"><strong>Known future conditions</strong><span>Optional</span></legend>
        <label className="toggle-row"><input type="checkbox" checked={settings.promotion} onChange={(event) => update("promotion", event.target.checked)} /><span>Planned promotion</span></label>
        <div className="condition-grid">
          <label><span>Discount (%)</span><input type="number" min="0" max="100" value={settings.discount} disabled={!settings.promotion} onChange={(event) => update("discount", Number(event.target.value))} /></label>
          <label><span>Selling price (VND)</span><input type="number" min="0" value={settings.price} onChange={(event) => update("price", Number(event.target.value))} /></label>
          <label><span>Promotion start</span><input type="date" disabled={!settings.promotion} /></label>
          <label><span>Promotion end</span><input type="date" disabled={!settings.promotion} /></label>
          <label className="condition-note"><span>Special event note</span><textarea placeholder="e.g. School holiday or local festival" /></label>
        </div>
      </fieldset>
      <div className="forecast-form__actions">
        <div aria-live="polite">{state === "loading" && <span className="run-state">Preparing daily predictions…</span>}{state === "success" && <span className="run-state run-state--success">Forecast generated successfully.</span>}{state === "error" && <span className="run-state run-state--error">Forecast generation failed. Please try again.</span>}</div>
        <div><button type="button" className="forecast-button" onClick={onReset}>Reset</button><button type="submit" className="forecast-button forecast-button--primary" disabled={state === "loading"}>{state === "loading" ? "Generating…" : "Generate Forecast"}</button></div>
      </div>
    </form>
  </section>;
};
