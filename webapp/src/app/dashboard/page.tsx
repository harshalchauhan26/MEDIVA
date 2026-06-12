import Link from "next/link";
import { redirect } from "next/navigation";
import { AlertTriangle, CalendarCheck, Clock3, Pill, TrendingUp } from "lucide-react";
import { prisma } from "@/lib/db";
import { getSession } from "@/lib/auth";
import { LOW_STOCK_THRESHOLD, isExpiringSoon } from "@/lib/api";
import { BarChart, Donut } from "@/components/charts";
import RecoveryTrends from "@/components/RecoveryTrends";

export const dynamic = "force-dynamic";

const DAY = 24 * 60 * 60 * 1000;
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function startOfDay(d: Date): Date {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x;
}

export default async function DashboardPage() {
  const session = getSession();
  if (!session) redirect("/login");

  const now = new Date();
  const today = startOfDay(now);
  const weekStart = new Date(today.getTime() - today.getDay() * DAY);
  const last30 = new Date(today.getTime() - 29 * DAY);

  const [upcoming, weekAppts, createdRecent, medicines] = await Promise.all([
    prisma.appointment.findMany({
      where: { startsAt: { gte: now }, status: { not: "CANCELLED" } },
      include: { patient: { select: { name: true } }, doctor: { include: { user: { select: { name: true } } } } },
      orderBy: { startsAt: "asc" },
      take: 6,
    }),
    prisma.appointment.findMany({
      where: { startsAt: { gte: weekStart, lt: new Date(weekStart.getTime() + 7 * DAY) } },
      select: { startsAt: true, status: true },
    }),
    prisma.appointment.findMany({
      where: { createdAt: { gte: last30 } },
      select: { createdAt: true },
    }),
    prisma.medicine.findMany({ select: { quantity: true, expiryDate: true, price: true } }),
  ]);

  // Stat cards
  const confirmedUpcoming = await prisma.appointment.count({
    where: { startsAt: { gte: now }, status: "CONFIRMED" },
  });
  const pendingUpcoming = await prisma.appointment.count({
    where: { startsAt: { gte: now }, status: "PENDING" },
  });
  const activeReservations = await prisma.reservation.count({ where: { status: "PENDING" } });

  const lowStock = medicines.filter((m) => m.quantity < LOW_STOCK_THRESHOLD).length;
  const expiring = medicines.filter((m) => isExpiringSoon(m.expiryDate)).length;
  const inStock = medicines.filter((m) => m.quantity > 0).length;
  const criticalAlerts = lowStock + expiring;
  const inventoryValue = medicines.reduce((sum, m) => sum + Number(m.price) * m.quantity, 0);

  // Week appointments bar chart (by weekday)
  const weekBuckets = Array(7).fill(0);
  for (const a of weekAppts) weekBuckets[new Date(a.startsAt).getDay()]++;

  // Today's appointments by 2-hour bucket (12 buckets)
  const todayBuckets = Array(12).fill(0);
  for (const a of weekAppts) {
    const d = new Date(a.startsAt);
    if (d >= today && d < new Date(today.getTime() + DAY)) todayBuckets[Math.floor(d.getHours() / 2)]++;
  }
  const todayCount = todayBuckets.reduce((a, b) => a + b, 0);

  // Activity trend series from createdAt
  const dayBuckets = Array(12).fill(0); // today, 2h buckets
  const weekSeries = Array(7).fill(0); // last 7 days
  const monthSeries = Array(30).fill(0); // last 30 days
  for (const a of createdRecent) {
    const d = new Date(a.createdAt);
    if (d >= today) dayBuckets[Math.floor(d.getHours() / 2)]++;
    const dayIndexFromStart = Math.floor((startOfDay(d).getTime() - last30.getTime()) / DAY);
    if (dayIndexFromStart >= 0 && dayIndexFromStart < 30) monthSeries[dayIndexFromStart]++;
    const weekIndex = Math.floor((startOfDay(d).getTime() - (today.getTime() - 6 * DAY)) / DAY);
    if (weekIndex >= 0 && weekIndex < 7) weekSeries[weekIndex]++;
  }

  const stats = [
    { label: "Confirmed Appointments", value: confirmedUpcoming, icon: CalendarCheck, hint: `${pendingUpcoming} pending`, tone: "green" as const },
    { label: "Active Reservations", value: activeReservations, icon: Pill, hint: "for pickup", tone: "green" as const },
    { label: "Critical Alerts", value: criticalAlerts, icon: AlertTriangle, hint: `${lowStock} low · ${expiring} expiring`, tone: "red" as const },
  ];

  const hour = now.getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";
  const firstName = session.name.split(" ")[0];

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div>
        <h2 className="text-xl font-semibold sm:text-2xl">
          {greeting}, {firstName}
        </h2>
        <p className="text-sm text-slate-500">
          Here&apos;s what&apos;s happening across MediVa today.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {stats.map((s) => (
          <div key={s.label} className="card flex items-start justify-between p-5">
            <div>
              <p className="text-sm text-slate-500">{s.label}</p>
              <p className="mt-2 text-3xl font-semibold">{String(s.value).padStart(2, "0")}</p>
              <p className={`mt-1 text-xs ${s.tone === "red" ? "text-red-600" : "text-brand-600"}`}>{s.hint}</p>
            </div>
            <div className={`rounded-xl p-2.5 ${s.tone === "red" ? "bg-red-50 text-red-600" : "bg-brand-50 text-brand-600"}`}>
              <s.icon className="h-5 w-5" />
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Upcoming events */}
        <div className="card p-5 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-semibold">Upcoming Appointments</h3>
            <Link href="/doctors" className="text-xs font-medium text-brand-600 hover:underline">View all</Link>
          </div>
          {upcoming.length === 0 ? (
            <p className="py-8 text-center text-sm text-slate-400">No upcoming appointments.</p>
          ) : (
            <ul className="space-y-3">
              {upcoming.map((a) => (
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
                    <p className="truncate text-sm font-medium">{a.patient.name}</p>
                    <p className="truncate text-xs text-slate-500">
                      {a.doctor.user.name} · {a.doctor.specialty}
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

        {/* Inventory health donut */}
        <div className="card flex flex-col items-center justify-center p-5">
          <h3 className="self-start font-semibold">Inventory Health</h3>
          <Donut value={inStock} max={medicines.length} label="in stock" size={140} />
          <div className="mt-2 grid w-full grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-lg bg-slate-50 p-2">
              <p className="font-semibold text-slate-900">{medicines.length}</p>
              <p className="text-slate-400">SKUs</p>
            </div>
            <div className="rounded-lg bg-slate-50 p-2">
              <p className="font-semibold text-slate-900">
                ₹{inventoryValue.toLocaleString("en-IN")}
              </p>
              <p className="text-slate-400">stock value</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Week appointments bar */}
        <div className="card p-5 lg:col-span-2">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <h3 className="font-semibold">Appointments This Week</h3>
              <p className="text-xs text-slate-400">{weekAppts.length} total scheduled</p>
            </div>
            <TrendingUp className="h-5 w-5 text-brand-500" />
          </div>
          <BarChart data={weekBuckets} labels={WEEKDAYS} height={150} />
        </div>

        {/* Today small widget */}
        <div className="card p-5">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Today</h3>
            <Clock3 className="h-4 w-4 text-slate-400" />
          </div>
          <p className="mt-2 text-3xl font-semibold">{todayCount}</p>
          <p className="text-xs text-slate-400">appointments today</p>
          <div className="mt-3">
            <BarChart data={todayBuckets} labels={todayBuckets.map(() => "")} height={90} />
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <RecoveryTrends series={{ day: dayBuckets, week: weekSeries, month: monthSeries }} />
        </div>
        <div className="card flex flex-col justify-center gap-3 p-5 lg:col-span-2">
          <h3 className="font-semibold">Quick actions</h3>
          <div className="grid gap-3 sm:grid-cols-3">
            <Link href="/doctors" className="rounded-xl border border-slate-200 p-4 transition hover:-translate-y-0.5 hover:border-brand-300 hover:shadow-card">
              <CalendarCheck className="h-5 w-5 text-brand-600" />
              <p className="mt-2 text-sm font-medium">Book appointment</p>
            </Link>
            <Link href="/pharmacy" className="rounded-xl border border-slate-200 p-4 transition hover:-translate-y-0.5 hover:border-brand-300 hover:shadow-card">
              <Pill className="h-5 w-5 text-brand-600" />
              <p className="mt-2 text-sm font-medium">Check stock</p>
            </Link>
            <Link href="/chat" className="rounded-xl border border-slate-200 p-4 transition hover:-translate-y-0.5 hover:border-brand-300 hover:shadow-card">
              <TrendingUp className="h-5 w-5 text-brand-600" />
              <p className="mt-2 text-sm font-medium">Ask MediVa AI</p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
