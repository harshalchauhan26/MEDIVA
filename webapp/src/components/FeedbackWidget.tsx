"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { CheckCircle2, Loader2, MessageSquarePlus, Star, X } from "lucide-react";

/**
 * Floating "Give feedback" widget for real-user evaluation. Bottom-left so it
 * never collides with the MediVa chat launcher (bottom-right). Submissions land
 * in the Feedback table and are readable by admins via GET /api/feedback.
 */
export default function FeedbackWidget() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [message, setMessage] = useState("");
  const [email, setEmail] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [error, setError] = useState("");

  // Hide on auth screens to keep them focused.
  if (pathname === "/login" || pathname === "/signup" || pathname === "/onboarding") return null;

  function reset() {
    setRating(0);
    setHover(0);
    setMessage("");
    setEmail("");
    setState("idle");
    setError("");
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!rating) {
      setError("Please pick a rating.");
      return;
    }
    setState("loading");
    setError("");
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating, message, page: pathname, email }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Could not send feedback.");
      }
      setState("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send feedback.");
      setState("error");
    }
  }

  return (
    <>
      {open && (
        <div className="fixed bottom-24 left-4 z-50 w-[min(22rem,calc(100vw-2rem))] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl">
          <div className="flex items-center justify-between bg-slate-900 px-4 py-3 text-white">
            <div>
              <p className="text-sm font-semibold">Share your feedback</p>
              <p className="text-xs text-slate-300">Help us improve MediVa</p>
            </div>
            <button
              onClick={() => {
                setOpen(false);
                if (state === "done") reset();
              }}
              aria-label="Close feedback"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {state === "done" ? (
            <div className="flex flex-col items-center gap-2 px-5 py-8 text-center">
              <CheckCircle2 className="h-10 w-10 text-brand-600" />
              <p className="text-sm font-medium">Thank you!</p>
              <p className="text-xs text-slate-500">Your feedback was recorded.</p>
              <button
                className="btn-secondary mt-2"
                onClick={() => {
                  setOpen(false);
                  reset();
                }}
              >
                Done
              </button>
            </div>
          ) : (
            <form onSubmit={submit} className="space-y-3 p-4">
              <div>
                <p className="mb-1 text-sm font-medium text-slate-600">How was your experience?</p>
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button
                      key={n}
                      type="button"
                      aria-label={`${n} star${n > 1 ? "s" : ""}`}
                      onClick={() => setRating(n)}
                      onMouseEnter={() => setHover(n)}
                      onMouseLeave={() => setHover(0)}
                      className="p-0.5"
                    >
                      <Star
                        className={`h-7 w-7 transition ${
                          n <= (hover || rating)
                            ? "fill-amber-400 text-amber-400"
                            : "text-slate-300"
                        }`}
                      />
                    </button>
                  ))}
                </div>
              </div>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="What worked well? What was confusing or broken?"
                rows={3}
                className="input resize-none"
                required
              />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email (optional, for follow-up)"
                className="input"
              />
              {error && <p className="text-sm text-red-600">{error}</p>}
              <button
                type="submit"
                className="btn-primary w-full justify-center"
                disabled={state === "loading"}
              >
                {state === "loading" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <MessageSquarePlus className="h-4 w-4" />
                )}
                Send feedback
              </button>
            </form>
          )}
        </div>
      )}

      <button
        onClick={() => setOpen((v) => !v)}
        aria-label="Give feedback"
        className="fixed bottom-6 left-4 z-50 flex items-center gap-2 rounded-full bg-white px-4 py-3 text-sm font-medium text-slate-700 shadow-lg ring-1 ring-slate-200 transition hover:bg-slate-50"
      >
        {open ? <X className="h-5 w-5" /> : <MessageSquarePlus className="h-5 w-5 text-brand-600" />}
        <span className="hidden sm:inline">{open ? "Close" : "Feedback"}</span>
      </button>
    </>
  );
}
