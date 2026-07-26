import { useMemo, useState } from "react";
import { demandStatus, formatDate, summarizeForecast, type ForecastPoint } from "../forecastData";

type SortKey = "date" | "predictedUnitsSold" | "projectedInventory";
const PAGE_SIZE = 10;

export const ForecastTable = ({ data }: { data: ForecastPoint[] }) => {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("date");
  const [ascending, setAscending] = useState(true);
  const [page, setPage] = useState(1);
  const summary = summarizeForecast(data);
  const rows = useMemo(() => data
    .filter((point) => point.date.includes(query) || formatDate(point.date).toLowerCase().includes(query.toLowerCase()))
    .toSorted((a, b) => {
      const difference = sort === "date" ? a.date.localeCompare(b.date) : a[sort] - b[sort];
      return ascending ? difference : -difference;
    }), [data, query, sort, ascending]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const currentPage = Math.min(page, pages);
  const shown = rows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);
  const changeSort = (key: SortKey) => { if (sort === key) setAscending(current => !current); else { setSort(key); setAscending(true); } };

  return <section className="forecast-section" aria-labelledby="daily-table-heading">
    <div className="forecast-section__heading"><div><span className="forecast-index">05</span><h2 id="daily-table-heading">Daily forecast</h2><p>Inspect every daily prediction and its operational context.</p></div><label className="forecast-search"><span>Search by date</span><input type="search" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="YYYY-MM-DD or month" /></label></div>
    <div className="forecast-table-scroll"><table><thead><tr>
      <th><button type="button" onClick={() => changeSort("date")}>Date {sort === "date" ? ascending ? "↑" : "↓" : ""}</button></th>
      <th><button type="button" onClick={() => changeSort("predictedUnitsSold")}>Predicted units {sort === "predictedUnitsSold" ? ascending ? "↑" : "↓" : ""}</button></th>
      <th>Lower bound</th><th>Upper bound</th><th>Planned price</th><th>Discount</th><th>Promotion</th>
      <th><button type="button" onClick={() => changeSort("projectedInventory")}>Projected inventory {sort === "projectedInventory" ? ascending ? "↑" : "↓" : ""}</button></th><th>Demand status</th>
    </tr></thead><tbody>{shown.map(point => {
      const status = demandStatus(point.predictedUnitsSold, summary.average, summary.peak.predictedUnitsSold);
      return <tr key={point.date} className={status === "Peak" ? "peak-row" : ""}><td><strong>{formatDate(point.date, true)}</strong><small>{point.date}</small></td><td>{point.predictedUnitsSold.toLocaleString()}</td><td>{point.lowerBound.toLocaleString()}</td><td>{point.upperBound.toLocaleString()}</td><td>₫{point.plannedSellingPrice.toLocaleString()}</td><td>{point.discountPct}%</td><td><span className={`forecast-badge ${point.isPromoted ? "forecast-badge--warning" : "forecast-badge--neutral"}`}>{point.isPromoted ? "Promoted" : "Standard"}</span></td><td className={point.projectedInventory < 0 ? "negative-value" : ""}>{point.projectedInventory.toLocaleString()}</td><td><span className={`demand-status demand-status--${status.toLowerCase()}`}>{status}</span></td></tr>;
    })}</tbody></table></div>
    <div className="table-pagination"><span>Showing {rows.length ? (currentPage - 1) * PAGE_SIZE + 1 : 0}–{Math.min(currentPage * PAGE_SIZE, rows.length)} of {rows.length} days</span><div><button type="button" className="forecast-button" disabled={currentPage === 1} onClick={() => setPage(current => current - 1)}>Previous</button><span>Page {currentPage} of {pages}</span><button type="button" className="forecast-button" disabled={currentPage === pages} onClick={() => setPage(current => current + 1)}>Next</button></div></div>
  </section>;
};
