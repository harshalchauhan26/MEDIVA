import { redirect } from "next/navigation";
import { MessageSquareText, Star } from "lucide-react";
import { prisma } from "@/lib/db";
import { getSession } from "@/lib/auth";

export const dynamic = "force-dynamic";

export default async function AdminFeedbackPage() {
  const session = getSession();
  if (!session) redirect("/login");
  if (session.role !== "ADMIN") redirect("/dashboard");

  const items = await prisma.feedback.findMany({
    orderBy: { createdAt: "desc" },
    take: 200,
  });

  const avg =
    items.length > 0
      ? (items.reduce((sum, f) => sum + f.rating, 0) / items.length).toFixed(1)
      : "—";

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="card p-5">
          <p className="text-sm text-slate-500">Total responses</p>
          <p className="mt-2 text-3xl font-semibold">{items.length}</p>
        </div>
        <div className="card p-5">
          <p className="text-sm text-slate-500">Average rating</p>
          <p className="mt-2 flex items-center gap-2 text-3xl font-semibold">
            {avg}
            <Star className="h-6 w-6 fill-amber-400 text-amber-400" />
          </p>
        </div>
        <div className="card p-5">
          <p className="text-sm text-slate-500">With contact email</p>
          <p className="mt-2 text-3xl font-semibold">{items.filter((f) => f.email).length}</p>
        </div>
      </div>

      <div className="card p-5">
        <div className="mb-4 flex items-center gap-2">
          <MessageSquareText className="h-5 w-5 text-brand-600" />
          <h3 className="font-semibold">Evaluation feedback</h3>
        </div>
        {items.length === 0 ? (
          <p className="py-10 text-center text-sm text-slate-400">No feedback yet.</p>
        ) : (
          <ul className="divide-y divide-slate-100">
            {items.map((f) => (
              <li key={f.id} className="py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-0.5">
                    {[1, 2, 3, 4, 5].map((n) => (
                      <Star
                        key={n}
                        className={`h-4 w-4 ${
                          n <= f.rating ? "fill-amber-400 text-amber-400" : "text-slate-200"
                        }`}
                      />
                    ))}
                  </div>
                  <span className="text-xs text-slate-400">
                    {new Date(f.createdAt).toLocaleString("en-IN")}
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-700">{f.message}</p>
                <div className="mt-1 flex flex-wrap gap-x-3 text-xs text-slate-400">
                  {f.page && <span>on {f.page}</span>}
                  {f.email && <span>· {f.email}</span>}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
