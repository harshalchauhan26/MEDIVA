import Link from "next/link";
import { redirect } from "next/navigation";
import {
  ArrowRight,
  Bot,
  Boxes,
  CalendarCheck,
  CalendarClock,
  Clock3,
  Pill,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import { prisma } from "@/lib/db";
import { getSession } from "@/lib/auth";
import { LOW_STOCK_THRESHOLD, isExpiringSoon } from "@/lib/api";
import { Donut } from "@/components/charts";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const session = getSession();
  if (!session) redirect("/login");

  const now = new Date();
  const hour = now.getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";
  const firstName = session.name.split(" ")[0];
  const isStaff = session.role === "PHARMACIST" || session.role === "ADMIN";

  // Appointments relevant to *this* user.
  const apptWhere =
    session.role === "DOCTOR"
      ? { doctor: { userId: session.userId }, startsAt: { gte: now }, status: { not: "CANCELLED" as const } }
      : { patientId: session.userId, startsAt: { gte: now }, status: { not: "CANCELLED" as const } };

  const [medicines, doctorsAvailable, myUpcoming] = await Promise.all([
    prisma.medicine.findMany({ select: { quantity: true, price: true, expiryDate: true } }),
    prisma.doctor.count({ where: { status: "AVAILABLE" } }),
    prisma.appointment.findMany({
      where: apptWhere,
      include: {
        patient: { select: { name: true } },
        doctor: { include: { user: { select: { name: true } } } },
      },
      orderBy: { startsAt: "asc" },
      take: 4,
    }),
  ]);

  const inStock = medicines.filter((m) => m.quantity > 0).length;
  const lowStock = medicines.filter((m) => m.quantity < LOW_STOCK_THRESHOLD).length;
  const expiring = medicines.filter((m) => isExpiringSoon(m.expiryDate)).length;
  const inventoryValue = medicines.reduce((sum, m) => sum + Number(m.price) * m.quantity, 0);

  const quickActions = [
    { href: "/doctors", icon: CalendarCheck, title: "Book an appointment", desc: "Find a doctor and pick a live slot" },
    { href: "/pharmacy", icon: Pill, title: "Browse pharmacy", desc: "Search stock, prices, reserve for pickup" },
    { href: "/chat", icon: Bot, title: "Ask MediVa AI", desc: "24/7 medical answers and help" },
    ...(isStaff
      ? [{ href: "/admin/inventory", icon: Boxes, title: "Manage inventory", desc: "Add, edit, and track stock alerts" }]
      : []),
  ];

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* Hero */}
      <section className="card overflow-hidden">
        <div className="relative bg-gradient-to-br from-brand-600 to-brand-800 p-6 text-white sm:p-9">
          <div className="flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-xs w-fit">
            <Stethoscope className="h-4 w-4" /> MediVa Health Platform
          </div>
          <h1 className="mt-4 text-2xl font-semibold sm:text-3xl">
            {greeting}, {firstName} 👋
          </h1>
          <p className="mt-2 max-w-xl text-sm text-brand-50/90">
            Book appointments, check live pharmacy stock, and get instant answers from MediVa —
            your 24/7 AI health assistant.
          </p>
          <div className="mt-5 flex flex-wrap gap-3">
            <Link
              href="/doctors"
              className="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-sm font-medium text-brand-700 transition hover:bg-brand-50"
            >
              <CalendarCheck className="h-4 w-4" /> Book appointment
            </Link>
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 rounded-xl border border-white/40 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/10"
            >
              <Bot className="h-4 w-4" /> Talk to MediVa
            </Link>
          </div>
        </div>
      </section>

      {/* At-a-glance stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="card flex items-center justify-between p-5">
          <div>
            <p className="text-sm text-slate-500">Medicines in stock</p>
            <p className="mt-2 text-3xl font-semibold">{inStock}</p>
            <p className="mt-1 text-xs text-brand-600">{medicines.length} total SKUs</p>
          </div>
          <div className="rounded-xl bg-brand-50 p-2.5 text-brand-600">
            <Pill className="h-5 w-5" />
          </div>
        </div>
        <div className="card flex items-center justify-between p-5">
          <div>
            <p className="text-sm text-slate-500">Doctors available</p>
            <p className="mt-2 text-3xl font-semibold">{doctorsAvailable}</p>
            <p className="mt-1 text-xs text-brand-600">ready to see you</p>
          </div>
          <div className="rounded-xl bg-brand-50 p-2.5 text-brand-600">
            <Stethoscope className="h-5 w-5" />
          </div>
        </div>
        <div className="card flex items-center justify-between p-5">
          <div>
            <p className="text-sm text-slate-500">Your upcoming visits</p>
            <p className="mt-2 text-3xl font-semibold">{myUpcoming.length}</p>
            <p className="mt-1 text-xs text-slate-400">
              {myUpcoming.length === 0 ? "none booked yet" : "scheduled"}
            </p>
          </div>
          <div className="rounded-xl bg-brand-50 p-2.5 text-brand-600">
            <CalendarClock className="h-5 w-5" />
          </div>
        </div>
      </div>

      {/* Quick actions */}
      <section>
        <h2 className="mb-3 text-sm font-semibold text-slate-500">Quick actions</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {quickActions.map((a) => (
            <Link
              key={a.href}
              href={a.href}
              className="card-interactive group flex items-start gap-4 p-5"
            >
              <div className="rounded-xl bg-brand-50 p-3 text-brand-600">
                <a.icon className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="flex items-center gap-1 font-medium">
                  {a.title}
                  <ArrowRight className="h-4 w-4 -translate-x-1 opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100" />
                </p>
                <p className="mt-1 text-sm text-slate-500">{a.desc}</p>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Upcoming + inventory */}
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-semibold">Your upcoming appointments</h3>
            <Link href="/doctors" className="text-xs font-medium text-brand-600 hover:underline">
              {session.role === "DOCTOR" ? "My schedule" : "Book new"}
            </Link>
          </div>
          {myUpcoming.length === 0 ? (
            <div className="flex flex-col items-center gap-3 py-10 text-center">
              <div className="rounded-2xl bg-brand-50 p-3 text-brand-600">
                <CalendarClock className="h-6 w-6" />
              </div>
              <div>
                <p className="text-sm font-medium">No appointments yet</p>
                <p className="text-xs text-slate-400">Book your first visit in a couple of taps.</p>
              </div>
              <Link href="/doctors" className="btn-primary">
                <CalendarCheck className="h-4 w-4" /> Book an appointment
              </Link>
            </div>
          ) : (
            <ul className="space-y-3">
              {myUpcoming.map((a) => (
                <li key={a.id} className="flex items-center gap-3">
                  <div className="flex w-14 shrink-0 flex-col items-center rounded-lg bg-slate-50 py-1.5">
                    <span className="text-xs font-semibold text-brand-700">
                      {new Date(a.startsAt).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {new Date(a.startsAt).toLocaleDateString("en-IN", { day: "2-digit", month: "short" })}
                    </span>
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">
                      {session.role === "DOCTOR" ? a.patient.name : a.doctor.user.name}
                    </p>
                    <p className="truncate text-xs text-slate-500">
                      {session.role === "DOCTOR" ? "Patient" : a.doctor.specialty}
                    </p>
                  </div>
                  <span className={`badge ${a.status === "CONFIRMED" ? "badge-green" : "badge-amber"}`}>
                    {a.status.toLowerCase()}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card flex flex-col items-center justify-center p-5">
          <div className="mb-1 flex w-full items-center justify-between">
            <h3 className="font-semibold">Pharmacy health</h3>
            {(lowStock > 0 || expiring > 0) && (
              <span className="badge badge-amber flex items-center gap-1">
                <Clock3 className="h-3 w-3" /> {lowStock + expiring} alerts
              </span>
            )}
          </div>
          <Donut value={inStock} max={medicines.length} label="in stock" size={130} />
          <div className="mt-3 grid w-full grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-lg bg-slate-50 p-2">
              <p className="font-semibold text-slate-900">₹{inventoryValue.toLocaleString("en-IN")}</p>
              <p className="text-slate-400">stock value</p>
            </div>
            <div className="rounded-lg bg-slate-50 p-2">
              <p className="font-semibold text-slate-900">{lowStock + expiring}</p>
              <p className="text-slate-400">need attention</p>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <section className="card flex items-center gap-3 p-5 text-sm text-slate-500">
        <ShieldCheck className="h-5 w-5 shrink-0 text-brand-600" />
        MediVa is a reference and learning tool, not a replacement for professional medical advice,
        diagnosis, or treatment. In an emergency, contact your local emergency services.
      </section>
    </div>
  );
}
