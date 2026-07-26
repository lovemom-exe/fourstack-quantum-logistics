export type Horizon = 14 | 30 | 60 | 90;

export interface ForecastPoint {
  date: string;
  predictedUnitsSold: number;
  lowerBound: number;
  upperBound: number;
  plannedSellingPrice: number;
  discountPct: number;
  isPromoted: boolean;
  projectedInventory: number;
}

export interface ForecastSummaryData {
  total: number;
  average: number;
  peak: ForecastPoint;
  lowest: ForecastPoint;
  trendPct: number;
  confidence: number;
}

export const FORECAST_PRODUCTS = [
  { id: "PRD-001", name: "Fresh Milk 1L", category: "Dairy", stock: 18_500 },
  { id: "PRD-002", name: "Greek Yogurt 500g", category: "Dairy", stock: 7_840 },
  { id: "PRD-003", name: "Baby Spinach 250g", category: "Fresh produce", stock: 3_120 },
  { id: "PRD-004", name: "Salmon Fillet 400g", category: "Seafood", stock: 1_760 },
  { id: "all", name: "All products", category: "Portfolio", stock: 68_400 },
];

const toIsoDate = (date: Date) =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
export const formatDate = (date: string, short = false) =>
  new Intl.DateTimeFormat("en-US", short ? { month: "short", day: "numeric" } : { month: "long", day: "numeric", year: "numeric" }).format(new Date(`${date}T00:00:00`));

export const generateForecastData = (horizon: Horizon, startDate: string, promotion = true, discountPct = 10, price = 31_500): ForecastPoint[] => {
  let inventory = 18_500;
  const start = new Date(`${startDate}T00:00:00`);
  return Array.from({ length: horizon }, (_, index) => {
    const date = new Date(start);
    date.setDate(start.getDate() + index);
    const promoted = promotion && index >= Math.floor(horizon * .25) && index <= Math.floor(horizon * .45);
    const weekly = Math.sin(index * Math.PI / 3.5) * 34;
    const trend = index * 1.7;
    const peak = index === Math.floor(horizon * .55) || index === Math.floor(horizon * .78) ? 120 : 0;
    const predicted = Math.max(0, Math.round(350 + weekly + trend + peak + (promoted ? 55 : 0)));
    const spread = Math.round(42 + predicted * .08);
    inventory -= predicted;
    return {
      date: toIsoDate(date), predictedUnitsSold: predicted,
      lowerBound: Math.max(0, predicted - spread), upperBound: predicted + spread,
      plannedSellingPrice: promoted ? Math.round(price * (1 - discountPct / 100)) : price,
      discountPct: promoted ? discountPct : 0, isPromoted: promoted,
      projectedInventory: Math.round(inventory),
    };
  });
};

export const summarizeForecast = (data: ForecastPoint[]): ForecastSummaryData => {
  const total = data.reduce((sum, item) => sum + item.predictedUnitsSold, 0);
  const peak = data.reduce((current, item) => item.predictedUnitsSold > current.predictedUnitsSold ? item : current);
  const lowest = data.reduce((current, item) => item.predictedUnitsSold < current.predictedUnitsSold ? item : current);
  const comparisonWindow = Math.max(1, Math.floor(data.length / 4));
  const first = data.slice(0, comparisonWindow).reduce((sum, item) => sum + item.predictedUnitsSold, 0) / comparisonWindow;
  const last = data.slice(-comparisonWindow).reduce((sum, item) => sum + item.predictedUnitsSold, 0) / comparisonWindow;
  return { total, average: Math.round(total / data.length), peak, lowest, trendPct: Math.round(((last - first) / first) * 100), confidence: 88 };
};

export const demandStatus = (value: number, average: number, peak: number) => {
  if (value === peak) return "Peak";
  if (value >= average * 1.15) return "High";
  if (value <= average * .85) return "Low";
  return "Normal";
};

export const calculateInventoryImpact = (data: ForecastPoint[]) => {
  const current = 18_500, incoming = 3_200;
  const demand = data.reduce((sum, point) => sum + point.predictedUnitsSold, 0);
  const remaining = current + incoming - demand;
  const stockout = data.find((point) => point.projectedInventory + incoming < 0);
  const shortage = Math.max(0, -remaining);
  return { current, incoming, demand, remaining, stockoutDate: stockout?.date, shortage, reorder: Math.max(0, shortage + Math.round(demand * .15)) };
};

export const toForecastCsv = (data: ForecastPoint[]) => {
  const header = "date,predicted_units_sold,lower_bound,upper_bound,planned_selling_price,discount_pct,is_promoted,projected_inventory";
  return [header, ...data.map((item) => [item.date, item.predictedUnitsSold, item.lowerBound, item.upperBound, item.plannedSellingPrice, item.discountPct, item.isPromoted, item.projectedInventory].join(","))].join("\n");
};
