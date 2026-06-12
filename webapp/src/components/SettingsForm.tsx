"use client";

import { useState } from "react";
import { AlertTriangle, Check, Loader2, Save, Trash2, UserCog } from "lucide-react";

type Initial = { name: string; email: string; phone: string; role: string };

export default function SettingsForm({ initial }: { initial: Initial }) {
  const [name, setName] = useState(initial.name);
  const [phone, setPhone] = useState(initial.phone);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");

  async function saveDetails(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setSaved(false);
    setError("");
    try {
      const res = await fetch("/api/auth/profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, phone }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not save changes.");
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save changes.");
    } finally {
      setSaving(false);
    }
  }

  async function deleteAccount() {
    setDeleting(true);
    setDeleteError("");
    try {
      const res = await fetch("/api/auth/account", { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Could not delete account.");
      }
      window.location.href = "/login";
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Could not delete account.");
      setDeleting(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Account details */}
      <section className="card p-6">
        <div className="flex items-center gap-2">
          <UserCog className="h-5 w-5 text-brand-600" />
          <h2 className="text-lg font-semibold">Account details</h2>
        </div>
        <p className="mt-1 text-sm text-slate-500">Update your name and contact number.</p>

        <form onSubmit={saveDetails} className="mt-5 space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-600">Full name</label>
            <input
              required
              minLength={2}
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-600">Email</label>
            <input value={initial.email} disabled className="input cursor-not-allowed bg-slate-50 text-slate-500" />
            <p className="mt-1 text-xs text-slate-400">Email can&apos;t be changed.</p>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-slate-600">
              Mobile number <span className="font-normal text-slate-400">(WhatsApp confirmations)</span>
            </label>
            <input
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+91 98765 43210"
              className="input"
              inputMode="tel"
            />
          </div>
          {error && <p className="text-sm text-red-600">{error}</p>}
          <div className="flex items-center gap-3">
            <button type="submit" className="btn-primary" disabled={saving}>
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              Save changes
            </button>
            {saved && (
              <span className="flex items-center gap-1 text-sm text-brand-600">
                <Check className="h-4 w-4" /> Saved
              </span>
            )}
          </div>
        </form>
      </section>

      {/* Danger zone */}
      <section className="card border-red-200 p-6">
        <div className="flex items-center gap-2 text-red-600">
          <AlertTriangle className="h-5 w-5" />
          <h2 className="text-lg font-semibold">Delete account</h2>
        </div>
        <p className="mt-1 text-sm text-slate-500">
          Permanently delete your account and all associated data — appointments, reservations, and
          profile. This cannot be undone.
        </p>

        {!confirmOpen ? (
          <button
            onClick={() => setConfirmOpen(true)}
            className="mt-4 inline-flex items-center gap-2 rounded-xl border border-red-300 bg-white px-4 py-2 text-sm font-medium text-red-600 transition hover:bg-red-50"
          >
            <Trash2 className="h-4 w-4" /> Delete my account
          </button>
        ) : (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50/50 p-4">
            <p className="text-sm text-slate-700">
              Type <span className="font-semibold">DELETE</span> to confirm.
            </p>
            <input
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder="DELETE"
              className="input mt-2"
            />
            {deleteError && <p className="mt-2 text-sm text-red-600">{deleteError}</p>}
            <div className="mt-3 flex gap-2">
              <button
                onClick={deleteAccount}
                disabled={confirmText !== "DELETE" || deleting}
                className="inline-flex items-center gap-2 rounded-xl bg-red-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                Permanently delete
              </button>
              <button
                onClick={() => {
                  setConfirmOpen(false);
                  setConfirmText("");
                  setDeleteError("");
                }}
                className="btn-secondary"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
