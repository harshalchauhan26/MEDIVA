"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { addDays, format, isSameDay } from "date-fns";
import { Loader2 } from "lucide-react";

type Props = {
  doctorId: string;
  selectedSlot: string | null;
  onSelectSlot: (iso: string | null) => void;
};

async function fetchSlots(doctorId: string, date: string): Promise<string[]> {
  const response = await fetch(`/api/doctors/${doctorId}/slots?date=${date}`);
  if (!response.ok) throw new Error("Failed to load slots.");
  const data = await response.json();
  return data.slots;
}

/** Interactive 14-day calendar strip with live open-slot grid for one doctor. */
export default function AppointmentCalendar({ doctorId, selectedSlot, onSelectSlot }: Props) {
  const days = useMemo(() => Array.from({ length: 14 }, (_, i) => addDays(new Date(), i)), []);
  const [selectedDay, setSelectedDay] = useState<Date>(days[0]);

  const dateParam = format(selectedDay, "yyyy-MM-dd");
  const { data: slots, isLoading } = useQuery({
    queryKey: ["slots", doctorId, dateParam],
    queryFn: () => fetchSlots(doctorId, dateParam),
    refetchInterval: 15_000,
  });

  return (
    <div className="space-y-4">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {days.map((day) => {
          const active = isSameDay(day, selectedDay);
          return (
            <button
              key={day.toISOString()}
              onClick={() => {
                setSelectedDay(day);
                onSelectSlot(null);
              }}
              className={`flex min-w-[3.5rem] flex-col items-center rounded-lg border px-2 py-2 text-xs transition ${
                active
                  ? "border-brand-600 bg-brand-600 text-white"
                  : "border-slate-200 bg-white text-slate-600 hover:border-brand-300"
              }`}
            >
              <span className="font-medium">{format(day, "EEE")}</span>
              <span className="text-base font-semibold">{format(day, "d")}</span>
              <span>{format(day, "MMM")}</span>
            </button>
          );
        })}
      </div>

      {isLoading ? (
        <div className="flex justify-center py-6">
          <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
        </div>
      ) : (slots ?? []).length === 0 ? (
        <p className="py-4 text-sm text-slate-500">
          No open slots on {format(selectedDay, "EEEE, MMM d")}. Try another day.
        </p>
      ) : (
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
          {(slots ?? []).map((iso) => {
            const active = selectedSlot === iso;
            return (
              <button
                key={iso}
                onClick={() => onSelectSlot(active ? null : iso)}
                className={`rounded-lg border px-2 py-2 text-sm transition ${
                  active
                    ? "border-brand-600 bg-brand-600 text-white"
                    : "border-slate-200 bg-white text-slate-700 hover:border-brand-400"
                }`}
              >
                {format(new Date(iso), "h:mm a")}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
