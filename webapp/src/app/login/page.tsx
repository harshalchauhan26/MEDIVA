"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Loader2, LogIn } from "lucide-react";

const GOOGLE_ERRORS: Record<string, string> = {
  google_not_configured: "Google sign-in is not configured on this server yet.",
  google_failed: "Google sign-in failed. Please try again.",
};

const DEMO_ACCOUNTS = [
  { label: "Patient", email: "patient@mediva.dev" },
  { label: "Doctor", email: "dr.rao@mediva.dev" },
  { label: "Pharmacist", email: "pharma@mediva.dev" },
  { label: "Admin", email: "admin@mediva.dev" },
];

const ROLE_HOME: Record<string, string> = {
  PATIENT: "/doctors",
  DOCTOR: "/doctor/dashboard",
  PHARMACIST: "/admin/inventory",
  ADMIN: "/admin/inventory",
};

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Surface ?error=... returned by the Google OAuth routes.
  useEffect(() => {
    const code = new URLSearchParams(window.location.search).get("error");
    if (code) setError(GOOGLE_ERRORS[code] ?? "Sign-in failed.");
  }, []);

  async function login(e?: React.FormEvent) {
    e?.preventDefault();
    setError("");
    setIsLoading(true);
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Login failed.");
      router.push(ROLE_HOME[data.role] ?? "/");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
      setIsLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-md space-y-6">
      <div className="card p-8">
        <h1 className="text-2xl font-semibold">Log in to MEDIVA</h1>
        <p className="mt-1 text-sm text-slate-500">Access booking, reservations, and dashboards.</p>

        <form onSubmit={login} className="mt-6 space-y-4">
          <input
            type="email"
            required
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="input"
          />
          <input
            type="password"
            required
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="input"
          />
          {error && <p className="text-sm text-red-600">{error}</p>}
          <button type="submit" className="btn-primary w-full justify-center" disabled={isLoading}>
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogIn className="h-4 w-4" />}
            Log in
          </button>
        </form>

        <p className="mt-5 text-center text-sm text-slate-500">
          New to MediVa?{" "}
          <Link href="/signup" className="font-medium text-brand-600 hover:underline">
            Create an account
          </Link>
        </p>
      </div>

      <div className="card p-6">
        <p className="text-sm font-medium">Demo accounts</p>
        <p className="mt-1 text-xs text-slate-500">
          Seeded by <code>npm run db:seed</code>. Password for all: <code>mediva123</code>
        </p>
        <div className="mt-3 grid grid-cols-2 gap-2">
          {DEMO_ACCOUNTS.map((account) => (
            <button
              key={account.email}
              onClick={() => {
                setEmail(account.email);
                setPassword("mediva123");
              }}
              className="btn-secondary justify-center text-xs"
            >
              {account.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
