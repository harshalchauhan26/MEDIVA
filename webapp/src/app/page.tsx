import Link from "next/link";
import { redirect } from "next/navigation";
import { Bot, CalendarClock, Pill, ShieldCheck, Stethoscope } from "lucide-react";
import { getSession } from "@/lib/auth";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const session = getSession();
  if (session) redirect("/dashboard");

  const features = [
    { icon: Bot, title: "MediVa AI Assistant", description: "Medical answers from our reference library, live stock checks, and appointment booking — 24/7." },
    { icon: Pill, title: "Pharmacy Inventory", description: "Real-time stock, dosage, ₹ pricing, expiry alerts, and online buy links. Reserve for pickup." },
    { icon: CalendarClock, title: "Appointment Booking", description: "Filter doctors by specialty and rating, pick a live slot, get WhatsApp confirmations." },
  ];

  return (
    <div className="space-y-10">
      <section className="card overflow-hidden">
        <div className="flex flex-col items-start gap-4 bg-gradient-to-br from-brand-700 to-brand-900 p-8 text-white sm:p-12">
          <div className="flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-xs">
            <Stethoscope className="h-4 w-4" /> Healthcare provider portal
          </div>
          <h1 className="max-w-2xl text-3xl font-semibold sm:text-4xl">
            Care, inventory, and answers in one place.
          </h1>
          <p className="max-w-2xl text-brand-100">
            MediVa Health combines a live pharmacy, doctor scheduling, and an AI assistant grounded
            in indexed medical references and aware of your live database.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/signup" className="rounded-xl bg-white px-4 py-2 text-sm font-medium text-brand-700 transition hover:bg-brand-50">
              Sign up free
            </Link>
            <Link href="/login" className="rounded-xl border border-white/40 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/10">
              Log in
            </Link>
            <Link href="/chat" className="rounded-xl border border-white/40 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/10">
              Talk to MediVa
            </Link>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {features.map((f) => (
          <div key={f.title} className="card-interactive p-6">
            <f.icon className="h-8 w-8 text-brand-600" />
            <h2 className="mt-4 font-semibold">{f.title}</h2>
            <p className="mt-2 text-sm text-slate-500">{f.description}</p>
          </div>
        ))}
      </section>

      <section className="card flex items-center gap-3 p-5 text-sm text-slate-500">
        <ShieldCheck className="h-5 w-5 shrink-0 text-brand-600" />
        MediVa is a reference and learning tool, not a replacement for professional medical advice,
        diagnosis, or treatment.
      </section>
    </div>
  );
}
