"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Clock3, ExternalLink, Loader2, Search } from "lucide-react";
import { formatINR } from "@/lib/format";
import type { BuyLink } from "@/lib/buylinks";

type Medicine = {
  id: string;
  name: string;
  genericName: string;
  dosage: string;
  quantity: number;
  price: number;
  locationShelf: string;
  expiryDate: string;
  lowStock: boolean;
  expiringSoon: boolean;
  buyLinks: BuyLink[];
};

type Tier = { label: string; badge: string };

function stockTier(m: Medicine): Tier {
  if (m.quantity === 0) return { label: "Out of stock", badge: "badge-red" };
  if (m.quantity < 10) return { label: "Critical", badge: "badge-red" };
  if (m.quantity < 25) return { label: "Low Stock", badge: "badge-amber" };
  return { label: "In Stock", badge: "badge-green" };
}

async function fetchMedicines(q: string): Promise<Medicine[]> {
  const response = await fetch(`/api/medicines?q=${encodeURIComponent(q)}`);
  if (!response.ok) throw new Error("Failed to load inventory.");
  return (await response.json()).medicines;
}

export default function PharmacyPage() {
  const [query, setQuery] = useState("");
  const [message, setMessage] = useState("");
  const queryClient = useQueryClient();

  // Pick up ?q= from the top-bar global search.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("q");
    if (q) setQuery(q);
  }, []);

  const { data: medicines, isLoading } = useQuery({
    queryKey: ["medicines", query],
    queryFn: () => fetchMedicines(query),
    refetchInterval: 20_000,
  });

  const reserve = useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`/api/medicines/${id}/reserve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ quantity: 1 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Reservation failed.");
      return data.reservation as { medicine: string };
    },
    onSuccess: (reservation) => {
      setMessage(`Reserved 1 × ${reservation.medicine}. A WhatsApp confirmation was triggered.`);
      queryClient.invalidateQueries({ queryKey: ["medicines"] });
    },
    onError: (error: Error) => setMessage(error.message),
  });

  const list = medicines ?? [];
  const criticalCount = list.filter((m) => m.quantity < 10).length;
  const expiringCount = list.filter((m) => m.expiringSoon).length;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm text-slate-500">Real-time stock monitoring · Central Pharmacy</p>
        </div>
        <div className="flex gap-2">
          <span className="badge badge-amber"><AlertTriangle className="h-3 w-3" /> {criticalCount} low / critical</span>
          <span className="badge badge-red"><Clock3 className="h-3 w-3" /> {expiringCount} expiring soon</span>
        </div>
      </div>

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search medicines, batches, or shelf locations…"
          className="input pl-9"
        />
      </div>

      {message && (
        <p className="rounded-xl border border-brand-100 bg-brand-50 px-4 py-2 text-sm text-brand-700">{message}</p>
      )}

      {isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {list.map((medicine) => {
            const tier = stockTier(medicine);
            return (
              <div key={medicine.id} className="card flex flex-col p-5">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h2 className="font-semibold leading-tight">{medicine.name}</h2>
                    <p className="text-xs text-slate-400">{medicine.genericName}</p>
                  </div>
                  <span className={`badge ${tier.badge}`}>{tier.label}</span>
                </div>

                <p className="mt-3 text-xs uppercase tracking-wide text-slate-400">Current stock</p>
                <div className="flex items-end gap-1">
                  <span className="text-3xl font-semibold">{medicine.quantity}</span>
                  <span className="mb-1 text-xs text-slate-400">units</span>
                </div>

                <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-500">
                  <div>
                    <p className="text-slate-400">Unit price</p>
                    <p className="font-medium text-slate-900">{formatINR(medicine.price)}</p>
                  </div>
                  <div>
                    <p className="text-slate-400">Expiry</p>
                    <p className={`font-medium ${medicine.expiringSoon ? "text-orange-600" : "text-slate-900"}`}>
                      {medicine.expiryDate.slice(0, 10)}
                    </p>
                  </div>
                  <div className="col-span-2">
                    <p className="text-slate-400">Shelf · {medicine.locationShelf}</p>
                    <p className="text-slate-500">{medicine.dosage}</p>
                  </div>
                </div>

                <button
                  onClick={() => reserve.mutate(medicine.id)}
                  disabled={medicine.quantity === 0 || reserve.isPending}
                  className="btn-primary mt-4 justify-center text-xs"
                >
                  Reserve for Pickup
                </button>

                <div className="mt-3 border-t border-slate-100 pt-3">
                  <p className="mb-2 text-xs font-medium text-slate-400">Buy online</p>
                  <div className="flex flex-wrap gap-1.5">
                    {medicine.buyLinks.map((link) => (
                      <a
                        key={link.app}
                        href={link.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-600 transition hover:border-brand-300 hover:text-brand-700"
                      >
                        {link.app}
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
          {list.length === 0 && (
            <p className="flex items-center gap-2 text-sm text-slate-500">
              <AlertTriangle className="h-4 w-4" /> No medicines match your search.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
