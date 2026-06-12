"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Repeat, ShieldCheck, Stethoscope, User, Boxes } from "lucide-react";

const ROLES = [
  { label: "Patient", email: "patient@mediva.dev", icon: User, desc: "Book appointments, reserve medicines, chat with MediVa.", home: "/dashboard" },
  { label: "Doctor", email: "dr.rao@mediva.dev", icon: Stethoscope, desc: "Manage your schedule, confirm appointments, set status.", home: "/doctor/dashboard" },
  { label: "Pharmacist", email: "pharma@mediva.dev", icon: Boxes, desc: "Add/update stock, track batches, expiry and low-stock alerts.", home: "/admin/inventory" },
  { label: "Admin", email: "admin@mediva.dev", icon: ShieldCheck, desc: "Full inventory control and clinic oversight.", home: "/admin/inventory" },
];

export default function RolesPage() {
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState("");

  async function switchTo(email: string, home: string) {
    setError("");
    setLoading(email);
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password: "mediva123" }),
      });
      if (!response.ok) throw new Error("Could not switch role.");
      router.push(home);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed.");
      setLoading(null);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Repeat className="h-5 w-5 text-brand-600" />
        <div>
          <h2 className="text-lg font-semibold">Role Switcher</h2>
          <p className="text-sm text-slate-500">
            Instantly sign in as a seeded demo account to explore each role&apos;s view.
          </p>
        </div>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="grid gap-4 sm:grid-cols-2">
        {ROLES.map((role) => (
          <button
            key={role.email}
            onClick={() => switchTo(role.email, role.home)}
            disabled={loading !== null}
            className="card flex items-start gap-4 p-5 text-left transition hover:border-brand-300 disabled:opacity-60"
          >
            <div className="rounded-xl bg-brand-50 p-3 text-brand-600">
              <role.icon className="h-5 w-5" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <p className="font-semibold">{role.label}</p>
                {loading === role.email && <Loader2 className="h-4 w-4 animate-spin text-slate-400" />}
              </div>
              <p className="mt-1 text-sm text-slate-500">{role.desc}</p>
              <p className="mt-2 text-xs text-slate-400">{role.email}</p>
            </div>
          </button>
        ))}
      </div>

      <p className="text-xs text-slate-400">All demo accounts use the password <code>mediva123</code>.</p>
    </div>
  );
}
