"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Clock3, Loader2, Minus, Plus, Trash2 } from "lucide-react";
import { formatINR } from "@/lib/format";

type Medicine = {
  id: string;
  name: string;
  genericName: string;
  sku: string;
  batchNumber: string;
  dosage: string;
  quantity: number;
  price: number;
  locationShelf: string;
  expiryDate: string;
  lowStock: boolean;
  expiringSoon: boolean;
};

const EMPTY_FORM = {
  name: "",
  genericName: "",
  sku: "",
  batchNumber: "",
  dosage: "",
  quantity: 0,
  price: 0,
  locationShelf: "",
  expiryDate: "",
};

export default function InventoryAdminPage() {
  const [query, setQuery] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [showForm, setShowForm] = useState(false);
  const [message, setMessage] = useState("");
  const queryClient = useQueryClient();

  const { data: medicines, isLoading, error } = useQuery<Medicine[]>({
    queryKey: ["medicines", query],
    queryFn: async () => {
      const response = await fetch(`/api/medicines?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error("Failed to load inventory.");
      return (await response.json()).medicines;
    },
    refetchInterval: 15_000,
  });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["medicines"] });

  const create = useMutation({
    mutationFn: async () => {
      const response = await fetch("/api/medicines", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          quantity: Number(form.quantity),
          price: Number(form.price),
          expiryDate: new Date(form.expiryDate).toISOString(),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Create failed. Are you logged in as pharmacist/admin?");
    },
    onSuccess: () => {
      setForm(EMPTY_FORM);
      setShowForm(false);
      setMessage("Medicine added.");
      invalidate();
    },
    onError: (e: Error) => setMessage(e.message),
  });

  const adjustQuantity = useMutation({
    mutationFn: async ({ id, quantity }: { id: string; quantity: number }) => {
      const response = await fetch(`/api/medicines/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quantity }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Update failed.");
    },
    onSuccess: invalidate,
    onError: (e: Error) => setMessage(e.message),
  });

  const remove = useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`/api/medicines/${id}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Delete failed.");
    },
    onSuccess: invalidate,
    onError: (e: Error) => setMessage(e.message),
  });

  const lowStockCount = (medicines ?? []).filter((m) => m.lowStock).length;
  const expiringCount = (medicines ?? []).filter((m) => m.expiringSoon).length;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Inventory Admin</h1>
          <p className="text-sm text-slate-500">
            Pharmacist/Admin dashboard — SKU, batch, expiry, and stock management.
          </p>
        </div>
        <button onClick={() => setShowForm((v) => !v)} className="btn-primary">
          <Plus className="h-4 w-4" /> Add medicine
        </button>
      </div>

      <div className="flex flex-wrap gap-3 text-sm">
        <span className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-amber-700">
          <AlertTriangle className="h-4 w-4" /> {lowStockCount} low stock (&lt;10 units)
        </span>
        <span className="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 px-3 py-2 text-orange-700">
          <Clock3 className="h-4 w-4" /> {expiringCount} expiring within 60 days
        </span>
      </div>

      {message && (
        <p className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700">{message}</p>
      )}

      {showForm && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            create.mutate();
          }}
          className="card grid gap-3 p-6 sm:grid-cols-2 lg:grid-cols-3"
        >
          {(
            [
              ["name", "Name"],
              ["genericName", "Generic name"],
              ["sku", "SKU"],
              ["batchNumber", "Batch number"],
              ["dosage", "Dosage"],
              ["locationShelf", "Shelf location"],
            ] as const
          ).map(([key, label]) => (
            <input
              key={key}
              required
              placeholder={label}
              value={form[key]}
              onChange={(e) => setForm({ ...form, [key]: e.target.value })}
              className="input"
            />
          ))}
          <input
            type="number"
            min={0}
            required
            placeholder="Quantity"
            value={form.quantity || ""}
            onChange={(e) => setForm({ ...form, quantity: Number(e.target.value) })}
            className="input"
          />
          <input
            type="number"
            min={0}
            step="0.01"
            required
            placeholder="Price"
            value={form.price || ""}
            onChange={(e) => setForm({ ...form, price: Number(e.target.value) })}
            className="input"
          />
          <input
            type="date"
            required
            value={form.expiryDate}
            onChange={(e) => setForm({ ...form, expiryDate: e.target.value })}
            className="input"
          />
          <button type="submit" className="btn-primary justify-center sm:col-span-2 lg:col-span-3" disabled={create.isPending}>
            {create.isPending && <Loader2 className="h-4 w-4 animate-spin" />} Save medicine
          </button>
        </form>
      )}

      <div className="max-w-md">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search inventory…"
          className="input"
        />
      </div>

      <div className="card overflow-x-auto">
        {isLoading ? (
          <div className="flex justify-center py-10">
            <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
          </div>
        ) : error ? (
          <p className="px-6 py-8 text-sm text-red-600">Failed to load inventory.</p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
              <tr>
                {["Medicine", "SKU / Batch", "Shelf", "Expiry", "Price", "Stock", ""].map((h) => (
                  <th key={h} className="px-4 py-3">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {(medicines ?? []).map((medicine) => (
                <tr key={medicine.id} className={medicine.lowStock ? "bg-amber-50/60" : undefined}>
                  <td className="px-4 py-3">
                    <p className="font-medium">{medicine.name}</p>
                    <p className="text-xs text-slate-500">{medicine.genericName} · {medicine.dosage}</p>
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-500">
                    {medicine.sku}
                    <br />
                    {medicine.batchNumber}
                  </td>
                  <td className="px-4 py-3">{medicine.locationShelf}</td>
                  <td className="px-4 py-3">
                    <span className={medicine.expiringSoon ? "font-medium text-orange-600" : ""}>
                      {medicine.expiryDate.slice(0, 10)}
                    </span>
                  </td>
                  <td className="px-4 py-3">{formatINR(medicine.price)}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <button
                        className="rounded border border-slate-300 p-1 hover:bg-slate-100"
                        onClick={() =>
                          adjustQuantity.mutate({ id: medicine.id, quantity: Math.max(0, medicine.quantity - 1) })
                        }
                      >
                        <Minus className="h-3 w-3" />
                      </button>
                      <span className={`min-w-8 text-center font-medium ${medicine.lowStock ? "text-amber-700" : ""}`}>
                        {medicine.quantity}
                      </span>
                      <button
                        className="rounded border border-slate-300 p-1 hover:bg-slate-100"
                        onClick={() => adjustQuantity.mutate({ id: medicine.id, quantity: medicine.quantity + 1 })}
                      >
                        <Plus className="h-3 w-3" />
                      </button>
                      {medicine.lowStock && (
                        <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700">Low</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <button
                      className="rounded p-1.5 text-red-500 hover:bg-red-50"
                      onClick={() => remove.mutate(medicine.id)}
                      aria-label={`Delete ${medicine.name}`}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
