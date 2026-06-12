"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import {
  Bot,
  Boxes,
  CalendarDays,
  HeartPulse,
  LayoutDashboard,
  LifeBuoy,
  LogOut,
  Menu,
  MessageSquareText,
  Pill,
  Repeat,
  Search,
  Settings,
  Stethoscope,
  X,
} from "lucide-react";
import type { Session } from "@/lib/auth";

type NavItem = { href: string; label: string; icon: typeof LayoutDashboard };

function navFor(role: Session["role"]): NavItem[] {
  const items: NavItem[] = [
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { href: "/pharmacy", label: "Pharmacy Stock", icon: Pill },
    { href: "/doctors", label: "Appointments", icon: CalendarDays },
  ];
  if (role === "DOCTOR") items.push({ href: "/doctor/dashboard", label: "My Schedule", icon: Stethoscope });
  if (role === "PHARMACIST" || role === "ADMIN")
    items.push({ href: "/admin/inventory", label: "Inventory Admin", icon: Boxes });
  if (role === "ADMIN")
    items.push({ href: "/admin/feedback", label: "User Feedback", icon: MessageSquareText });
  items.push({ href: "/chat", label: "MediVa AI", icon: Bot });
  items.push({ href: "/roles", label: "Role Switcher", icon: Repeat });
  return items;
}

const TITLES: Record<string, string> = {
  "/dashboard": "Dashboard",
  "/pharmacy": "Pharmacy Inventory",
  "/doctors": "Appointment Booking",
  "/doctor/dashboard": "My Schedule",
  "/admin/inventory": "Inventory Admin",
  "/admin/feedback": "User Feedback",
  "/chat": "MediVa AI Assistant",
  "/roles": "Role Switcher",
};

function GuestHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold text-brand-700">
          <HeartPulse className="h-6 w-6" /> MediVa Health
        </Link>
        <div className="flex items-center gap-2">
          <Link href="/login" className="btn-secondary">Log in</Link>
          <Link href="/signup" className="btn-primary">Sign up</Link>
        </div>
      </div>
    </header>
  );
}

export default function AppShell({
  session,
  children,
}: {
  session: Session | null;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [search, setSearch] = useState("");

  const bare = pathname === "/login" || pathname === "/onboarding";

  if (!session) {
    return (
      <>
        <GuestHeader />
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
      </>
    );
  }
  if (bare) return <main className="mx-auto max-w-3xl px-4 py-10">{children}</main>;

  const nav = navFor(session.role);
  const title = TITLES[pathname] ?? "MediVa Health";

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/login";
  }

  function submitSearch(e: React.FormEvent) {
    e.preventDefault();
    if (search.trim()) router.push(`/pharmacy?q=${encodeURIComponent(search.trim())}`);
  }

  const SidebarBody = (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-white">
          <HeartPulse className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-semibold leading-tight">MediVa Health</p>
          <p className="text-xs text-slate-400">Provider Portal</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        {nav.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setMobileOpen(false)}
              className={`nav-link ${active ? "nav-link-active" : ""}`}
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-1 px-3 pb-3">
        <Link href="/chat" className="flex items-center justify-center gap-2 rounded-xl bg-brand-600 px-3 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700">
          <Bot className="h-4 w-4" /> MediVa AI · 24/7
        </Link>
        <button className="nav-link w-full" type="button">
          <Settings className="h-4 w-4" /> Settings
        </button>
        <button className="nav-link w-full" type="button">
          <LifeBuoy className="h-4 w-4" /> Support
        </button>
      </div>

      <div className="border-t border-slate-100 p-3">
        <div className="flex items-center gap-3 rounded-xl px-2 py-2">
          <div className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-full bg-slate-200 text-sm font-semibold text-slate-600">
            {session.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={session.image} alt="" className="h-full w-full object-cover" />
            ) : (
              session.name.charAt(0).toUpperCase()
            )}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{session.name}</p>
            <p className="truncate text-xs capitalize text-slate-400">{session.role.toLowerCase()}</p>
          </div>
          <button onClick={logout} aria-label="Log out" className="text-slate-400 hover:text-slate-700">
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 border-r border-slate-200 bg-white lg:block">
        {SidebarBody}
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/40" onClick={() => setMobileOpen(false)} />
          <aside className="absolute left-0 top-0 h-full w-64 bg-white shadow-xl">{SidebarBody}</aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur">
          <button className="lg:hidden" onClick={() => setMobileOpen(true)} aria-label="Open menu">
            <Menu className="h-5 w-5" />
          </button>
          <h1 className="text-lg font-semibold">{title}</h1>
          <form onSubmit={submitSearch} className="relative ml-auto hidden max-w-xs flex-1 sm:block">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search medicines, stock…"
              className="input pl-9"
            />
          </form>
        </header>
        <main className="flex-1 p-4 sm:p-6">{children}</main>
      </div>

      {/* Mobile close button when drawer open */}
      {mobileOpen && (
        <button
          className="fixed right-4 top-4 z-[60] rounded-full bg-white p-2 shadow lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-label="Close menu"
        >
          <X className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}
