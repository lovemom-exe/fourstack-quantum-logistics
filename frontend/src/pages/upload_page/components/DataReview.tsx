import { useState } from "react";
import { DATASETS, formatFileSize, type DatasetDefinition } from "../datasets";
import type { DatasetKey, UploadedFile } from "./CsvUploadCard";

export const DatasetSummaryCard = ({ dataset, file, onRemove, onPreview }: { dataset: DatasetDefinition; file: UploadedFile | null; onRemove: () => void; onPreview: () => void }) => (
    <article className="summary-card">
        <div className="summary-card__top"><h3>{dataset.name}</h3><span className={`status-badge ${file ? "status-badge--success" : "status-badge--neutral"}`}>{file ? "Uploaded" : "Not uploaded"}</span></div>
        {file ? <>
            <strong className="file-name" title={file.name}>{file.name}</strong>
            <dl className="metadata-grid"><div><dt>Rows</dt><dd>{file.rows.toLocaleString()}</dd></div><div><dt>Columns</dt><dd>{dataset.columns.length}</dd></div><div><dt>File size</dt><dd>{formatFileSize(file.size)}</dd></div><div><dt>Uploaded</dt><dd>{file.uploadedAt}</dd></div></dl>
            <div className="validation-line"><span className="state-dot state-dot--success" />Validation passed with warnings</div>
            <div className="card-actions"><button type="button" className="data-button data-button--primary" onClick={onPreview}>Preview</button><button type="button" className="data-button">Replace file</button><button type="button" className="data-button data-button--danger" onClick={onRemove}>Remove</button></div>
        </> : <div className="empty-state"><span>—</span><p>No file is available for this dataset.</p></div>}
    </article>
);

const previews: Record<DatasetKey, { columns: [string, string, number, string][]; rows: string[][] }> = {
    products: { columns: [["product_id", "String", 0, "PRD-001"], ["product_name", "String", 0, "Fresh milk 1L"], ["category", "String", 0, "Dairy"], ["shelf_life_days", "Integer", 0, "12"], ["base_price", "Decimal", 2, "32000"]], rows: [["PRD-001", "Fresh milk 1L", "Dairy", "12", "32000"], ["PRD-002", "Greek yogurt", "Dairy", "21", "28500"], ["PRD-003", "Baby spinach", "Produce", "6", "19000"], ["PRD-004", "Salmon fillet", "Seafood", "4", "145000"], ["PRD-005", "Sourdough loaf", "Bakery", "3", "52000"]] },
    suppliers: { columns: [["supplier_id", "String", 0, "SUP-01"], ["supplier_name", "String", 0, "Mekong Fresh"], ["supplier_score", "Decimal", 1, "4.8"], ["lead_time_days", "Integer", 0, "2"], ["supplier_country", "String", 0, "Vietnam"]], rows: [["SUP-01", "Mekong Fresh", "4.8", "2", "Vietnam"], ["SUP-02", "Dalat Growers", "4.6", "1", "Vietnam"], ["SUP-03", "Pacific Foods", "4.1", "4", "Singapore"], ["SUP-04", "Good Grain Co.", "3.9", "3", "Thailand"], ["SUP-05", "Coastal Catch", "4.7", "1", "Vietnam"]] },
    sales: { columns: [["transaction_date", "Date", 0, "2026-07-20"], ["product_id", "String", 0, "PRD-001"], ["store_id", "String", 0, "STR-014"], ["units_sold", "Integer", 12, "48"], ["selling_price", "Decimal", 2, "31000"], ["discount_pct", "Decimal", 46, "5.0"], ["is_promoted", "Boolean", 0, "true"]], rows: [["2026-07-20", "PRD-001", "STR-014", "48", "31000", "5.0", "true"], ["2026-07-20", "PRD-002", "STR-008", "22", "28500", "0", "false"], ["2026-07-21", "PRD-003", "STR-014", "67", "17100", "10.0", "true"], ["2026-07-21", "PRD-004", "STR-003", "14", "145000", "0", "false"], ["2026-07-22", "PRD-005", "STR-011", "31", "46800", "10.0", "true"]] },
    inventory: { columns: [["product_id", "String", 0, "PRD-001"], ["warehouse_id", "String", 0, "WH-HCM-01"], ["current_inventory", "Integer", 0, "480"], ["reserved_inventory", "Integer", 0, "60"], ["expiration_date", "Date", 7, "2026-08-03"], ["batch_id", "String", 0, "B-260712"]], rows: [["PRD-001", "WH-HCM-01", "480", "60", "2026-08-03", "B-260712"], ["PRD-002", "WH-HCM-01", "240", "18", "2026-08-12", "B-260718"], ["PRD-003", "WH-HCM-01", "115", "12", "2026-07-29", "B-260724"], ["PRD-004", "WH-HCM-01", "82", "8", "2026-07-28", "B-260725"], ["PRD-005", "WH-HCM-01", "190", "20", "2026-07-27", "B-260725"]] },
};

export const DataPreviewTable = ({ selected, onSelect }: { selected: DatasetKey; onSelect: (key: DatasetKey) => void }) => {
    const data = previews[selected];
    return <section className="data-section" aria-labelledby="preview-heading">
        <div className="data-section__heading"><div><span className="section-index">07</span><h2 id="preview-heading">Data preview</h2><p>Inspect the first five records and inferred schema from each uploaded file.</p></div><label className="inline-select"><span>Dataset</span><select value={selected} onChange={(e) => onSelect(e.target.value as DatasetKey)}>{DATASETS.map(d => <option key={d.key} value={d.key}>{d.name}</option>)}</select></label></div>
        <div className="schema-strip">{data.columns.map(([name, type, missing, example]) => <article key={name}><strong>{name}</strong><span>{type}</span><small>{missing} missing · e.g. {example}</small></article>)}</div>
        <div className="table-scroll"><table><thead><tr><th>#</th>{data.columns.map(([name]) => <th key={name}>{name}</th>)}</tr></thead><tbody>{data.rows.map((row, index) => <tr key={index}><td>{index + 1}</td>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>)}</tbody></table></div>
        <p className="table-caption">Showing 5 of {selected === "sales" ? "184,293" : selected === "inventory" ? "4,218" : selected === "products" ? "1,248" : "84"} rows</p>
    </section>;
};

type MappingStatus = "Matched" | "Needs review" | "Missing" | "Generated automatically" | "Ignored";
const mappings: { feature: string; description: string; requirement: string; uploaded: string; status: MappingStatus; confidence: number }[] = [
    { feature: "transaction_date", description: "Date of the recorded sale", requirement: "Required", uploaded: "transaction_date", status: "Matched", confidence: 100 },
    { feature: "product_id", description: "Stable product identifier", requirement: "Required", uploaded: "product_sku", status: "Needs review", confidence: 74 },
    { feature: "units_sold", description: "Historical demand prediction target", requirement: "Required", uploaded: "qty_sold", status: "Matched", confidence: 96 },
    { feature: "selling_price", description: "Actual unit selling price", requirement: "Required", uploaded: "selling_price", status: "Matched", confidence: 100 },
    { feature: "discount_pct", description: "Applied discount percentage", requirement: "Recommended", uploaded: "discount", status: "Needs review", confidence: 68 },
    { feature: "promotion_duration", description: "Length of the promotion", requirement: "Optional", uploaded: "", status: "Generated automatically", confidence: 100 },
    { feature: "weather_code", description: "Optional external weather label", requirement: "Optional", uploaded: "", status: "Missing", confidence: 0 },
];
const statusClass = (status: string) => status.toLowerCase().replaceAll(" ", "-");

export const ColumnMappingTable = () => {
    const [rows, setRows] = useState(mappings);
    return <section className="data-section" aria-labelledby="mapping-heading">
        <div className="data-section__heading"><div><span className="section-index">08</span><h2 id="mapping-heading">Column mapping</h2><p>Connect uploaded sales columns to the application’s standard features.</p></div><div className="heading-actions"><button type="button" className="data-button data-button--quiet" onClick={() => setRows(mappings)}>Reset mappings</button><button type="button" className="data-button" onClick={() => setRows(current => current.map(row => row.status === "Needs review" ? { ...row, status: "Matched", confidence: 92 } : row))}>Auto-map columns</button><button type="button" className="data-button data-button--primary">Save mappings</button></div></div>
        <div className="mapping-table table-scroll"><table><thead><tr><th>Standard feature</th><th>Requirement</th><th>Uploaded CSV column</th><th>Match status</th><th>Confidence</th></tr></thead><tbody>{rows.map((row, index) => <tr key={row.feature}><td><strong>{row.feature}</strong><small>{row.description}</small></td><td><span className={`requirement requirement--${row.requirement.toLowerCase()}`}>{row.requirement}</span></td><td><select aria-label={`Uploaded column for ${row.feature}`} value={row.uploaded} onChange={e => setRows(current => current.map((item, itemIndex) => itemIndex === index ? { ...item, uploaded: e.target.value, status: e.target.value ? "Matched" : "Missing", confidence: e.target.value ? 100 : 0 } : item))}><option value="">Not mapped</option>{["transaction_date", "product_sku", "qty_sold", "selling_price", "discount", "is_promoted"].map(option => <option key={option}>{option}</option>)}</select></td><td><span className={`mapping-status mapping-status--${statusClass(row.status)}`}>{row.status}</span></td><td><div className="confidence"><span><i style={{ width: `${row.confidence}%` }} /></span><strong>{row.confidence}%</strong></div></td></tr>)}</tbody></table></div>
    </section>;
};

export const UnusedColumns = () => <section className="data-section" aria-labelledby="unused-heading"><div className="data-section__heading"><div><span className="section-index">09</span><h2 id="unused-heading">Unused columns</h2><p>Choose whether non-standard fields should be retained for forecasting.</p></div><span className="status-badge status-badge--neutral">3 columns</span></div><div className="unused-list">{["sales_channel", "cashier_id", "receipt_note"].map((column, index) => <label key={column}><div><code>{column}</code><small>{index === 0 ? "Example: retail" : index === 1 ? "Example: EMP-104" : "Example: loyalty redemption"}</small></div><select defaultValue="Ignore"><option>Ignore</option><option>Use as additional feature</option><option>Use as identifier</option></select></label>)}</div></section>;

const checks = [
    ["Missing required columns", "Passed", 0, "All required fields are present in the uploaded files.", "success"],
    ["Missing values", "Warning", 58, "Optional and target columns contain empty values.", "warning"],
    ["Duplicate rows", "Warning", 126, "Potential duplicate transactions were detected.", "warning"],
    ["Invalid dates", "Passed", 0, "All transaction dates use a recognized format.", "success"],
    ["Negative units sold", "Error", 12, "Demand values must be zero or greater.", "error"],
    ["Negative prices", "Passed", 0, "No negative selling or cost prices found.", "success"],
    ["Invalid discount values", "Warning", 4, "Discount must be between 0 and 100 percent.", "warning"],
    ["Invalid expiration dates", "Error", 7, "Expiration occurs before the inventory snapshot.", "error"],
    ["Unknown product IDs", "Warning", 19, "IDs do not match the product master dataset.", "warning"],
    ["Unknown supplier IDs", "Passed", 0, "All supplier references are recognized.", "success"],
];
export const ValidationPanel = () => <section className="data-section" aria-labelledby="validation-heading"><div className="data-section__heading"><div><span className="section-index">10</span><h2 id="validation-heading">Data validation</h2><p>Resolve blocking errors and review warnings before creating a forecast.</p></div><div className="validation-overview"><span><i className="state-dot state-dot--success" />4 passed</span><span><i className="state-dot state-dot--warning" />4 warnings</span><span><i className="state-dot state-dot--error" />2 errors</span></div></div><div className="validation-list">{checks.map(([name, result, rows, explanation, state]) => <article key={name}><span className={`check-icon check-icon--${state}`}>{state === "success" ? "✓" : state === "warning" ? "!" : "×"}</span><div><strong>{name}</strong><p>{explanation}</p></div><span className={`status-badge status-badge--${state}`}>{result}</span><span className="affected"><strong>{Number(rows).toLocaleString()}</strong> affected rows</span><button type="button" className="data-button" disabled={rows === 0}>View affected rows</button></article>)}</div></section>;

const features = ["day_of_week", "is_weekend", "month", "quarter", "week_of_year", "days_until_expiry", "price_change_pct", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_30", "demand_variability", "promotion_duration"];
export const GeneratedFeatures = () => <section className="data-section" aria-labelledby="features-heading"><div className="data-section__heading"><div><span className="section-index">11</span><h2 id="features-heading">Automatically generated features</h2><p>These time-series signals are derived during forecast preparation.</p></div><span className="status-badge status-badge--success">{features.length} enabled</span></div><div className="feature-grid">{features.map(feature => <div key={feature}><code>{feature}</code><span>Generated automatically</span></div>)}</div></section>;

export const ReadinessPanel = ({ ready, warehouseComplete, uploadedCount }: { ready: boolean; warehouseComplete: boolean; uploadedCount: number }) => {
    const score = ready ? 88 : Math.round((Number(warehouseComplete) + uploadedCount) / 5 * 70);
    return <section className="readiness-panel" aria-labelledby="readiness-heading"><div className="readiness-score"><div className="score-ring" style={{ "--score": `${score * 3.6}deg` } as React.CSSProperties}><span><strong>{score}</strong>/100</span></div><div><span className="section-index">12 · Dataset readiness</span><h2 id="readiness-heading">{ready ? "Ready with warnings" : "Not ready"}</h2><p>{ready ? "Required inputs are complete. Resolve validation errors for the most reliable forecast." : "Complete the missing required inputs before forecasting."}</p></div></div><div className="readiness-checks">{[["Warehouse setup", warehouseComplete ? "Complete" : "Missing"], ["Required datasets", `${uploadedCount}/4 uploaded`], ["Required columns", "5/5 mapped"], ["Validation status", "2 errors"], ["Generated features", "15 enabled"]].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><button type="button" className="data-button data-button--primary readiness-cta" disabled={!ready} onClick={() => { if (ready) window.location.href = "/forecast"; }}>Continue to Forecast <span>→</span></button></section>;
};
