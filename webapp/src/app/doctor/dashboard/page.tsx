"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { format, isToday } from "date-fns";
import { CalendarDays, Loader2, Users } from "lucide-react";

type Appointment = {
  id: string;
  startsAt: string;
  status: "PENDING" | "CONFIRMED" | "CANCELLED" | "COMPLETED";
  symptoms: string;
  patientName: string;
  patientEmail: string;
};

type DashboardData = {
  doctor: { id: string; name: string; specialty: string; status: string; rating: number };
  appointments: Appointment[];
};

const DOCTOR_STATUSES = ["AVAILABLE", "IN_SESSION", "ON_LEAVE"] as const;

const STATUS_BADGES: Record<Appointment["status"], string> = {
  PENDING: "bg-amber-100 text-amber-700",
  CONFIRMED: "bg-green-100 text-green-700",
  CANCELLED: "bg-slate-200 text-slate-500",
  COMPLETED: "bg-brand-100 text-brand-700",
};

export default function DoctorDashboard() {
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery<DashboardData>({
    queryKey: ["doctor-me"],
    queryFn: async () => {
      const response = await fetch("/api/doctors/me");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Failed to load dashboard.");
      return payload;
    },
    refetchInterval: 20_000,
  });

  const setStatus = useMutation({
    mutationFn: async (status: string) => {
      const response = await fetch("/api/doctors/me", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      if (!response.ok) throw new Error("Failed to update status.");
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["doctor-me"] }),
  });

  const setAppointmentStatus = useMutation({
    mutationFn: async ({ id, status }: { id: string; status: string }) => {
      const response = await fetch(`/api/appointments/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      if (!response.ok) throw new Error("Failed to update appointment.");
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["doctor-me"] }),
  });

  if (isLoading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
      </div>
    );
  }
  if (error || !data) {
    return (
      <p className="text-sm text-red-600">
        {error instanceof Error ? error.message : "Failed to load."} Log in with a doctor account
        (e.g. dr.rao@mediva.dev).
      </p>
    );
  }

  const todays = data.appointments.filter((a) => isToday(new Date(a.startsAt)));
  const patients = Array.from(new Set(data.appointments.map((a) => a.patientName)));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">{data.doctor.name}</h1>
          <p className="text-sm text-slate-500">{data.doctor.specialty} · rating {data.doctor.rating.toFixed(1)}</p>
        </div>
        <div className="flex gap-2">
          {DOCTOR_STATUSES.map((status) => (
            <button
              key={status}
              onClick={() => setStatus.mutate(status)}
              disabled={setStatus.isPending}
              className={`rounded-lg px-3 py-2 text-xs font-medium transition ${
                data.doctor.status === status
                  ? "bg-brand-600 text-white"
                  : "border border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
              }`}
            >
              {status.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <div className="card p-5">
          <p className="text-3xl font-semibold text-brand-700">{todays.length}</p>
          <p className="mt-1 flex items-center gap-1 text-sm text-slate-500">
            <CalendarDays className="h-4 w-4" /> Appointments today
          </p>
        </div>
        <div className="card p-5">
          <p className="text-3xl font-semibold text-brand-700">{data.appointments.length}</p>
          <p className="mt-1 text-sm text-slate-500">Upcoming (from today)</p>
        </div>
        <div className="card p-5">
          <p className="text-3xl font-semibold text-brand-700">{patients.length}</p>
          <p className="mt-1 flex items-center gap-1 text-sm text-slate-500">
            <Users className="h-4 w-4" /> Patients scheduled
          </p>
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="border-b border-slate-200 px-6 py-4 font-semibold">Schedule</div>
        {data.appointments.length === 0 ? (
          <p className="px-6 py-8 text-sm text-slate-500">No upcoming appointments.</p>
        ) : (
          <ul className="divide-y divide-slate-100">
            {data.appointments.map((appointment) => (
              <li key={appointment.id} className="flex flex-wrap items-center gap-4 px-6 py-4">
                <div className="min-w-36">
                  <p className="text-sm font-medium">
                    {format(new Date(appointment.startsAt), "EEE, MMM d")}
                  </p>
                  <p className="text-sm text-slate-500">{format(new Date(appointment.startsAt), "h:mm a")}</p>
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{appointment.patientName}</p>
                  <p className="truncate text-sm text-slate-500">{appointment.symptoms}</p>
                </div>
                <span className={`rounded-full px-2 py-0.5 text-xs ${STATUS_BADGES[appointment.status]}`}>
                  {appointment.status.toLowerCase()}
                </span>
                {appointment.status === "PENDING" && (
                  <div className="flex gap-2">
                    <button
                      className="btn-primary !py-1 text-xs"
                      onClick={() => setAppointmentStatus.mutate({ id: appointment.id, status: "CONFIRMED" })}
                    >
                      Confirm
                    </button>
                    <button
                      className="btn-secondary !py-1 text-xs"
                      onClick={() => setAppointmentStatus.mutate({ id: appointment.id, status: "CANCELLED" })}
                    >
                      Cancel
                    </button>
                  </div>
                )}
                {appointment.status === "CONFIRMED" && (
                  <button
                    className="btn-secondary !py-1 text-xs"
                    onClick={() => setAppointmentStatus.mutate({ id: appointment.id, status: "COMPLETED" })}
                  >
                    Mark completed
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
