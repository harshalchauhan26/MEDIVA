"use client";

import { useState } from "react";
import { Sparkline } from "@/components/charts";

type Series = { day: number[]; week: number[]; month: number[] };
const RANGES: { key: keyof Series; label: string }[] = [
  { key: "day", label: "Day" },
  { key: "week", label: "Week" },
  { key: "month", label: "Month" },
];

export default function RecoveryTrends({ series }: { series: Series }) {
  const [range, setRange] = useState<keyof Series>("week");
  const data = series[range];
  const total = data.reduce((a, b) => a + b, 0);

  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold">Activity Trends</h3>
          <p className="text-xs text-slate-400">Appointments booked across the selected window</p>
        </div>
        <div className="flex rounded-lg bg-slate-100 p-0.5 text-xs">
          {RANGES.map((r) => (
            <button
              key={r.key}
              onClick={() => setRange(r.key)}
              className={`rounded-md px-2.5 py-1 font-medium transition ${
                range === r.key ? "bg-white text-brand-700 shadow-sm" : "text-slate-500"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>
      <p className="mt-3 text-2xl font-semibold">{total}</p>
      <div className="mt-2">
        <Sparkline data={data} height={90} />
      </div>
    </div>
  );
}
