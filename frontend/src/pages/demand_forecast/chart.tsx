// "use client";
import "../../style/main/section.css";
import "../../style/elements/card.css";
import * as React from "react";

export const description = "A step area chart";

interface DataRow {
  month: string;
  data: number;
}

const WIDTH = 720;
const HEIGHT = 360;
const PADDING = { top: 28, right: 28, bottom: 48, left: 46 };
const maxValue = 240;

const point = (index: number, value: number, chartData: DataRow[]) => {
  const innerWidth = WIDTH - PADDING.left - PADDING.right;
  const innerHeight = HEIGHT - PADDING.top - PADDING.bottom;
  return {
    x: PADDING.left + (innerWidth / (chartData.length - 1)) * index,
    y: PADDING.top + innerHeight - (value / maxValue) * innerHeight,
  };
}

const stepPath = (chartData: DataRow[]) => {
  const points = chartData.map((item, index) => point(index, item.data, chartData));
  return points
    .map((p, index) => {
      if (index === 0) return `M ${p.x} ${p.y}`;
      //   const prev = points[index - 1];
      return `H ${p.x} V ${p.y}`;
    })
    .join(" ");
}

const areaPath = (chartData: DataRow[]) => {
  const points = chartData.map((item, index) => point(index, item.data, chartData));
  const baseY = HEIGHT - PADDING.bottom;
  return `${stepPath(chartData)} L ${points[points.length - 1].x} ${baseY} H ${points[0].x} Z`;
}

const ChartAreaStep = ({ chartData }: { chartData: DataRow[] }) => {
  const [activeIndex, setActiveIndex] = React.useState(1);
  const active = chartData[activeIndex];
  const activePoint = point(activeIndex, active.data, chartData);
  const xTicks = chartData.map((item, index) => ({ ...point(index, 0, chartData), label: item.month.slice(0, 3) }));
  const yTicks = [0, 60, 120, 180, 240];

  return (
    <div>
      <div>
        <div>
          <p className="bento-card__number" style={{ color: "var(--muted)" }}>Desktop</p>
          <h3 className="bento-card__title" style={{ color: "var(--ink)" }}>Step Area Chart</h3>
        </div>
        <div>
          <span className="bento-card__number" style={{ color: "var(--muted)" }}>{active.month}</span>
        </div>
      </div>

      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full overflow-visible" role="img" aria-label="Step area chart for desktop traffic">
        <defs>
          <pattern id="pixel-grid" width="16" height="16" patternUnits="userSpaceOnUse">
            <path d="M 16 0 L 0 0 0 16" fill="none" stroke="var(--muted)" strokeOpacity="0.12" strokeWidth="1" />
          </pattern>
        </defs>

        <rect x={PADDING.left} y={PADDING.top} width={WIDTH - PADDING.left - PADDING.right} height={HEIGHT - PADDING.top - PADDING.bottom} fill="url(#pixel-grid)" />

        {yTicks.map((tick) => {
          const y = point(0, tick, chartData).y;
          return (
            <g key={tick} className="text-muted-foreground">
              <line x1={PADDING.left} x2={WIDTH - PADDING.right} y1={y} y2={y} stroke="var(--muted)" strokeOpacity="0.32" strokeDasharray="8 8" />
              <text x={PADDING.left - 14} y={y + 4} textAnchor="end" fill="var(--ink)" className="bento-card__text" fontSize="5">{tick}</text>
            </g>
          );
        })}

        <path d={areaPath(chartData)} fill="var(--safety)" opacity="0.38" />
        <path d={stepPath(chartData)} fill="none" stroke="var(--safety)" strokeWidth="3" strokeLinejoin="miter" strokeLinecap="square" />

        {chartData.map((item, index) => {
          const p = point(index, item.data, chartData);
          //   const isActive = index === activeIndex;
          return (
            <g key={item.month} onMouseEnter={() => setActiveIndex(index)} onFocus={() => setActiveIndex(index)} tabIndex={0} className="cursor-pointer outline-none">
              <line x1={p.x} x2={p.x} y1={PADDING.top} y2={HEIGHT - PADDING.bottom} stroke="transparent" strokeWidth="46" />
              <rect x={p.x - 7} y={p.y - 7} width="10" height="10" fill={"var(--inl)"} stroke="none" strokeWidth="0" />
            </g>
          );
        })}

        {xTicks.map((tick) => (
          <text key={tick.label} x={tick.x} y={HEIGHT - 18} textAnchor="middle" fill="var(--ink)" className="bento-card__text" fontSize="5">{tick.label}</text>
        ))}

        <g transform={`translate(${Math.min(activePoint.x + 14, WIDTH - 170)} ${Math.max(activePoint.y - 62, 18)})`}>
          <rect width="152" height="46" fill="var(--safety)" stroke="var(--ink)" strokeWidth="2.5" />
          <text x="12" y="18" fontSize="10" fill="currentColor">{active.month}</text>
          <text x="12" y="34" fontSize="10" fill="currentColor">Desktop: {active.data}</text>
        </g>
      </svg>
    </div>
  );
}

export { ChartAreaStep }