"use client";

/** Tiny dependency-free SVG charts for the dashboard widgets. */

export function BarChart({
  data,
  labels,
  height = 120,
}: {
  data: number[];
  labels: string[];
  height?: number;
}) {
  const max = Math.max(1, ...data);
  const barW = 100 / (data.length * 1.6);
  const gap = barW * 0.6;
  return (
    <svg viewBox={`0 0 100 ${height}`} className="w-full" preserveAspectRatio="none" style={{ height }}>
      {data.map((value, i) => {
        const h = (value / max) * (height - 22);
        const x = i * (barW + gap) + gap;
        return (
          <g key={i}>
            <rect
              x={x}
              y={height - 18 - h}
              width={barW}
              height={Math.max(h, 1)}
              rx={1.5}
              className="fill-brand-500"
            />
            <text
              x={x + barW / 2}
              y={height - 6}
              textAnchor="middle"
              className="fill-slate-400"
              style={{ fontSize: 5 }}
            >
              {labels[i]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function Sparkline({
  data,
  height = 80,
  fill = true,
}: {
  data: number[];
  height?: number;
  fill?: boolean;
}) {
  if (data.length === 0) data = [0, 0];
  const max = Math.max(1, ...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const stepX = 100 / (data.length - 1 || 1);
  const points = data.map((v, i) => {
    const x = i * stepX;
    const y = height - 6 - ((v - min) / range) * (height - 14);
    return [x, y] as const;
  });
  const line = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${line} L100,${height} L0,${height} Z`;
  return (
    <svg viewBox={`0 0 100 ${height}`} className="w-full" preserveAspectRatio="none" style={{ height }}>
      {fill && <path d={area} className="fill-brand-100/60" />}
      <path d={line} className="stroke-brand-600" strokeWidth={1.6} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function Donut({
  value,
  max,
  label,
  size = 120,
}: {
  value: number;
  max: number;
  label?: string;
  size?: number;
}) {
  const pct = max <= 0 ? 0 : Math.min(1, value / max);
  const r = 40;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 100 100" style={{ width: size, height: size }}>
      <circle cx="50" cy="50" r={r} className="fill-none stroke-slate-100" strokeWidth={12} />
      <circle
        cx="50"
        cy="50"
        r={r}
        className="fill-none stroke-brand-500"
        strokeWidth={12}
        strokeLinecap="round"
        strokeDasharray={`${c * pct} ${c}`}
        transform="rotate(-90 50 50)"
      />
      <text x="50" y="48" textAnchor="middle" className="fill-slate-900 font-semibold" style={{ fontSize: 18 }}>
        {Math.round(pct * 100)}%
      </text>
      {label && (
        <text x="50" y="63" textAnchor="middle" className="fill-slate-400" style={{ fontSize: 8 }}>
          {label}
        </text>
      )}
    </svg>
  );
}
