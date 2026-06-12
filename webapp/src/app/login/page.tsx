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

        <a
          href="/api/auth/google"
          className="btn-secondary mt-6 w-full justify-center gap-3"
        >
          <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden>
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1Z" />
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23Z" />
            <path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84Z" />
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1A11 11 0 0 0 2.18 7.06l3.66 2.84C6.71 7.31 9.14 5.38 12 5.38Z" />
          </svg>
          Continue with Google
        </a>

        <div className="my-5 flex items-center gap-3 text-xs text-slate-400">
          <span className="h-px flex-1 bg-slate-200" /> or email & password
          <span className="h-px flex-1 bg-slate-200" />
        </div>

        <form onSubmit={login} className="space-y-4">
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
