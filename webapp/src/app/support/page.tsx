"use client";

import { useState } from "react";
import { CheckCircle2, LifeBuoy, Loader2, Send, Star } from "lucide-react";

export default function SupportPage() {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [message, setMessage] = useState("");
  const [email, setEmail] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "done">("idle");
  const [error, setError] = useState("");

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
        body: JSON.stringify({ rating, message, page: "/support", email }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Could not send your message.");
      }
      setState("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send your message.");
      setState("idle");
    }
  }

  return (
    <div className="mx-auto max-w-xl">
      <div className="card p-6 sm:p-8">
        <div className="flex items-center gap-2 text-brand-700">
          <LifeBuoy className="h-5 w-5" />
          <h1 className="text-xl font-semibold">Support &amp; feedback</h1>
        </div>
        <p className="mt-1 text-sm text-slate-500">
          Hit a problem or have a suggestion? Tell us below — the team reviews every message.
        </p>

        {state === "done" ? (
          <div className="flex flex-col items-center gap-2 py-12 text-center">
            <CheckCircle2 className="h-12 w-12 text-brand-600" />
            <p className="text-lg font-medium">Thank you!</p>
            <p className="text-sm text-slate-500">Your message has been sent to the team.</p>
            <button
              className="btn-secondary mt-3"
              onClick={() => {
                setState("idle");
                setRating(0);
                setMessage("");
                setEmail("");
              }}
            >
              Send another
            </button>
          </div>
        ) : (
          <form onSubmit={submit} className="mt-6 space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-600">
                How would you rate your experience?
              </label>
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
                      className={`h-8 w-8 transition ${
                        n <= (hover || rating) ? "fill-amber-400 text-amber-400" : "text-slate-300"
                      }`}
                    />
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-600">Your message</label>
              <textarea
                required
                rows={5}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Describe the issue, or share what would make MediVa better…"
                className="input resize-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-600">
                Email <span className="font-normal text-slate-400">(optional, for a reply)</span>
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="input"
              />
            </div>
            {error && <p className="text-sm text-red-600">{error}</p>}
            <button type="submit" className="btn-primary w-full justify-center" disabled={state === "loading"}>
              {state === "loading" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              Send message
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
