import "../../style/main/section.css";
import "../../style/elements/card.css";
import * as React from "react";

export const description = "A step area chart";

export interface ChartDataRow {
  month: string;
  data: number;
}

const HEIGHT = 360;
const PADDING = { top: 28, right: 28, bottom: 54, left: 58 };

const getMaxValue = (chartData: ChartDataRow[]) => {
  const highest = Math.max(...chartData.map((item) => item.data), 1);
  const magnitude = 10 ** Math.floor(Math.log10(highest));
  return Math.ceil(highest / magnitude) * magnitude;
};

const point = (index: number, value: number, chartData: ChartDataRow[], width: number, maxValue: number) => {
  const innerWidth = width - PADDING.left - PADDING.right;
  const innerHeight = HEIGHT - PADDING.top - PADDING.bottom;
  const divisor = Math.max(chartData.length - 1, 1);
  return {
    x: PADDING.left + (innerWidth / divisor) * index,
    y: PADDING.top + innerHeight - (value / maxValue) * innerHeight,
  };
};

const stepPath = (chartData: ChartDataRow[], width: number, maxValue: number) =>
  chartData.map((item, index) => point(index, item.data, chartData, width, maxValue))
    .map((item, index) => index === 0 ? `M ${item.x} ${item.y}` : `H ${item.x} V ${item.y}`)
    .join(" ");

const areaPath = (chartData: ChartDataRow[], width: number, maxValue: number) => {
  const points = chartData.map((item, index) => point(index, item.data, chartData, width, maxValue));
  const baseY = HEIGHT - PADDING.bottom;
  return `${stepPath(chartData, width, maxValue)} L ${points.at(-1)?.x ?? PADDING.left} ${baseY} H ${points[0]?.x ?? PADDING.left} Z`;
};

const labelInterval = (length: number) => length <= 14 ? 1 : length <= 30 ? 5 : 10;

const ChartAreaStep = ({ chartData }: { chartData: ChartDataRow[] }) => {
  const [activeIndex, setActiveIndex] = React.useState(0);
  React.useEffect(() => setActiveIndex(0), [chartData]);
  const width = Math.max(720, chartData.length * (chartData.length > 30 ? 16 : 24));
  const maxValue = getMaxValue(chartData);
  const active = chartData[Math.min(activeIndex, chartData.length - 1)] ?? { month: "", data: 0 };
  const activePoint = point(Math.min(activeIndex, chartData.length - 1), active.data, chartData, width, maxValue);
  const interval = labelInterval(chartData.length);
  const xTicks = chartData
    .map((item, index) => ({ ...point(index, 0, chartData, width, maxValue), label: item.month, index }))
    .filter((tick) => tick.index === 0 || tick.index === chartData.length - 1 || tick.index % interval === 0);
  const yTicks = Array.from({ length: 5 }, (_, index) => Math.round((maxValue / 4) * index));

  return (
    <div className="forecast-chart-scroll">
      <div className="forecast-chart-canvas" style={{ minWidth: `${width}px` }}>
        <div>
          <div>
            <p className="bento-card__number" style={{ color: "var(--muted)" }}>Daily forecast</p>
            <h3 className="bento-card__title" style={{ color: "var(--ink)" }}>Units sold</h3>
          </div>
          <div><span className="bento-card__number" style={{ color: "var(--muted)" }}>{active.month}</span></div>
        </div>
        <svg viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label="Daily predicted units sold">
          <defs>
            <pattern id="pixel-grid" width="16" height="16" patternUnits="userSpaceOnUse">
              <path d="M 16 0 L 0 0 0 16" fill="none" stroke="var(--muted)" strokeOpacity="0.12" strokeWidth="1" />
            </pattern>
          </defs>
          <rect x={PADDING.left} y={PADDING.top} width={width - PADDING.left - PADDING.right} height={HEIGHT - PADDING.top - PADDING.bottom} fill="url(#pixel-grid)" />
          {yTicks.map((tick) => {
            const y = point(0, tick, chartData, width, maxValue).y;
            return <g key={tick}><line x1={PADDING.left} x2={width - PADDING.right} y1={y} y2={y} stroke="var(--muted)" strokeOpacity="0.32" strokeDasharray="8 8" /><text x={PADDING.left - 14} y={y + 4} textAnchor="end" fill="var(--ink)" className="bento-card__text" fontSize="8">{tick}</text></g>;
          })}
          <text x="13" y={HEIGHT / 2} transform={`rotate(-90 13 ${HEIGHT / 2})`} textAnchor="middle" fill="var(--ink)" fontSize="9">Units sold</text>
          <path d={areaPath(chartData, width, maxValue)} fill="var(--safety)" opacity="0.38" />
          <path d={stepPath(chartData, width, maxValue)} fill="none" stroke="var(--safety)" strokeWidth={chartData.length > 30 ? "1.5" : "2.5"} strokeLinejoin="miter" strokeLinecap="square" />
          {chartData.map((item, index) => {
            const itemPoint = point(index, item.data, chartData, width, maxValue);
            return <g key={`${item.month}-${index}`} onMouseEnter={() => setActiveIndex(index)} onFocus={() => setActiveIndex(index)} tabIndex={0} className="cursor-pointer outline-none">
              <line x1={itemPoint.x} x2={itemPoint.x} y1={PADDING.top} y2={HEIGHT - PADDING.bottom} stroke="transparent" strokeWidth={Math.max(10, (width - PADDING.left - PADDING.right) / chartData.length)} />
              {chartData.length <= 30 && <rect x={itemPoint.x - 2} y={itemPoint.y - 2} width="4" height="4" fill="var(--ink)" />}
            </g>;
          })}
          {xTicks.map((tick) => <text key={`${tick.index}-${tick.label}`} x={tick.x} y={HEIGHT - 20} textAnchor={tick.index === 0 ? "start" : tick.index === chartData.length - 1 ? "end" : "middle"} fill="var(--ink)" className="bento-card__text" fontSize="7">{tick.label}</text>)}
          <g transform={`translate(${Math.min(activePoint.x + 14, width - 170)} ${Math.max(activePoint.y - 62, 18)})`}>
            <rect width="152" height="46" fill="var(--safety)" stroke="var(--ink)" strokeWidth="2.5" />
            <text x="12" y="18" fontSize="10" fill="currentColor">{active.month}</text>
            <text x="12" y="34" fontSize="10" fill="currentColor">Units sold: {active.data}</text>
          </g>
        </svg>
      </div>
    </div>
  );
};

export { ChartAreaStep };
