"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { format } from "date-fns";
import { CalendarCheck, Loader2, Paperclip, Star, Stethoscope, X } from "lucide-react";
import AppointmentCalendar from "@/components/AppointmentCalendar";

type Doctor = {
  id: string;
  name: string;
  specialty: string;
  bio: string;
  rating: number;
  status: "AVAILABLE" | "IN_SESSION" | "ON_LEAVE";
};

const STATUS_STYLES: Record<Doctor["status"], string> = {
  AVAILABLE: "badge-green",
  IN_SESSION: "badge-amber",
  ON_LEAVE: "badge-slate",
};

async function fetchDoctors(specialty: string, minRating: number): Promise<Doctor[]> {
  const params = new URLSearchParams({ specialty, minRating: String(minRating) });
  const response = await fetch(`/api/doctors?${params}`);
  if (!response.ok) throw new Error("Failed to load doctors.");
  return (await response.json()).doctors;
}

export default function DoctorsPage() {
  const [specialty, setSpecialty] = useState("All");
  const [minRating, setMinRating] = useState(0);
  const [selectedDoctor, setSelectedDoctor] = useState<Doctor | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [symptoms, setSymptoms] = useState("");
  const [recordName, setRecordName] = useState("");
  const [message, setMessage] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const queryClient = useQueryClient();

  const { data: me } = useQuery({
    queryKey: ["me"],
    queryFn: async () => (await fetch("/api/auth/me")).json(),
  });
  const patientName = me?.user?.name ?? "";

  const { data: doctors, isLoading } = useQuery({
    queryKey: ["doctors", specialty, minRating],
    queryFn: () => fetchDoctors(specialty, minRating),
  });

  const specialties = useMemo(() => {
    const all = new Set((doctors ?? []).map((d) => d.specialty));
    return ["All", ...Array.from(all).sort()];
  }, [doctors]);

  const book = useMutation({
    mutationFn: async () => {
      const note = recordName ? `${symptoms}\n[Attached record: ${recordName}]` : symptoms;
      const response = await fetch("/api/appointments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doctorId: selectedDoctor!.id, startsAt: selectedSlot, symptoms: note }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Booking failed.");
      return data.appointment as { doctorName: string; startsAt: string };
    },
    onSuccess: (appointment) => {
      setMessage({
        kind: "ok",
        text: `Booked with ${appointment.doctorName} on ${format(new Date(appointment.startsAt), "EEE, MMM d 'at' h:mm a")}. WhatsApp confirmation sent.`,
      });
      setModalOpen(false);
      setSelectedSlot(null);
      setSymptoms("");
      setRecordName("");
      queryClient.invalidateQueries({ queryKey: ["slots"] });
    },
    onError: (error: Error) => setMessage({ kind: "err", text: error.message }),
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <select value={specialty} onChange={(e) => setSpecialty(e.target.value)} className="input max-w-48">
          {specialties.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select value={minRating} onChange={(e) => setMinRating(Number(e.target.value))} className="input max-w-44">
          <option value={0}>Any rating</option>
          <option value={4.5}>4.5+ stars</option>
          <option value={4.7}>4.7+ stars</option>
        </select>
      </div>

      {message && (
        <p className={`rounded-xl border px-4 py-2 text-sm ${message.kind === "ok" ? "border-brand-200 bg-brand-50 text-brand-700" : "border-red-200 bg-red-50 text-red-700"}`}>
          {message.text}
        </p>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          {isLoading && (
            <div className="flex justify-center py-10">
              <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
            </div>
          )}
          {(doctors ?? []).map((doctor) => (
            <button
              key={doctor.id}
              onClick={() => {
                setSelectedDoctor(doctor);
                setSelectedSlot(null);
                setMessage(null);
              }}
              className={`card w-full p-5 text-left transition ${selectedDoctor?.id === doctor.id ? "border-brand-500 ring-2 ring-brand-100" : "hover:border-brand-300"}`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="font-semibold">{doctor.name}</h2>
                  <p className="flex items-center gap-1 text-sm text-slate-500">
                    <Stethoscope className="h-3.5 w-3.5" /> {doctor.specialty}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <span className="flex items-center gap-1 text-sm font-medium text-amber-600">
                    <Star className="h-4 w-4 fill-amber-400 text-amber-400" /> {doctor.rating.toFixed(1)}
                  </span>
                  <span className={`badge ${STATUS_STYLES[doctor.status]}`}>
                    {doctor.status.replace("_", " ").toLowerCase()}
                  </span>
                </div>
              </div>
              <p className="mt-2 text-sm text-slate-600">{doctor.bio}</p>
            </button>
          ))}
        </div>

        <div className="card h-fit p-6">
          {selectedDoctor ? (
            <div className="space-y-4">
              <h2 className="font-semibold">Open slots — {selectedDoctor.name}</h2>
              <AppointmentCalendar
                doctorId={selectedDoctor.id}
                selectedSlot={selectedSlot}
                onSelectSlot={(iso) => {
                  setSelectedSlot(iso);
                  if (iso) setModalOpen(true);
                }}
              />
            </div>
          ) : (
            <p className="text-sm text-slate-500">Select a doctor to see their live availability.</p>
          )}
        </div>
      </div>

      {/* Confirm Booking modal */}
      {modalOpen && selectedDoctor && selectedSlot && (
        <div className="modal-backdrop" onClick={() => setModalOpen(false)}>
          <div className="card w-full max-w-md overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between bg-brand-600 px-5 py-4 text-white">
              <div>
                <p className="font-semibold">Confirm Booking</p>
                <p className="text-xs text-brand-100">
                  {format(new Date(selectedSlot), "EEEE, MMM d · h:mm a")}
                </p>
              </div>
              <button onClick={() => setModalOpen(false)} aria-label="Close">
                <X className="h-5 w-5" />
              </button>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                book.mutate();
              }}
              className="space-y-4 p-5"
            >
              <div className="flex items-center gap-3 rounded-xl bg-slate-50 p-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-100 text-brand-700">
                  <Stethoscope className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium">{selectedDoctor.name}</p>
                  <p className="text-xs text-slate-500">{selectedDoctor.specialty}</p>
                </div>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">Patient Full Name</label>
                <input value={patientName} readOnly className="input bg-slate-50" />
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">Symptoms &amp; Notes</label>
                <textarea
                  required
                  minLength={3}
                  rows={3}
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  placeholder="Describe your symptoms or reason for the visit…"
                  className="input resize-none"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">Previous Records (optional)</label>
                <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-dashed border-slate-300 px-3 py-2.5 text-sm text-slate-500 hover:border-brand-300">
                  <Paperclip className="h-4 w-4" />
                  {recordName || "Attach medical history or lab results"}
                  <input
                    type="file"
                    className="hidden"
                    onChange={(e) => setRecordName(e.target.files?.[0]?.name ?? "")}
                  />
                </label>
              </div>

              <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={() => setModalOpen(false)} className="btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn-primary" disabled={book.isPending}>
                  {book.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <CalendarCheck className="h-4 w-4" />}
                  Confirm Booking
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
