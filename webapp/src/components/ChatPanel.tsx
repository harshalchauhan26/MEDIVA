"use client";

import { useEffect, useRef, useState } from "react";
import { Bot, Loader2, Send, ShieldAlert, User, Wrench } from "lucide-react";

type Message = { role: "user" | "assistant"; content: string };

const TOOL_LABELS: Record<string, string> = {
  search_medical_knowledge: "Searching medical reference documents…",
  check_medicine_stock: "Checking live pharmacy inventory…",
  find_available_slots: "Looking up open appointment slots…",
  book_appointment: "Booking the appointment…",
};

const SUGGESTIONS = [
  "What are common symptoms of anemia?",
  "Is paracetamol in stock?",
  "Find me an open slot with a cardiologist tomorrow.",
];

export default function ChatPanel({ compact = false }: { compact?: boolean }) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hi, I'm MediVa — your 24/7 health assistant. I can answer medical questions from our reference library, check live medicine stock, and help you book a doctor's appointment.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, toolStatus]);

  async function send(text = input) {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    const history: Message[] = [...messages, { role: "user", content: trimmed }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setIsLoading(true);
    setToolStatus(null);

    const appendToLast = (text: string) =>
      setMessages((current) => {
        const next = [...current];
        const last = next[next.length - 1];
        next[next.length - 1] = { ...last, content: last.content + text };
        return next;
      });

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.filter((m) => m.content.trim().length > 0),
        }),
      });
      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.error || "MediVa is unavailable right now.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let newline;
        while ((newline = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.type === "token") {
            setToolStatus(null);
            appendToLast(ev.text);
          } else if (ev.type === "tool") {
            setToolStatus(TOOL_LABELS[ev.name] ?? "Working…");
          } else if (ev.type === "error") {
            appendToLast(ev.message);
          }
        }
      }
    } catch (error) {
      appendToLast(error instanceof Error ? error.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
      setToolStatus(null);
    }
  }

  const hasUserMessage = messages.some((m) => m.role === "user");

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-800">
        <ShieldAlert className="h-4 w-4 shrink-0" />
        MediVa is an AI assistant, not a medical professional. For emergencies, call your local
        emergency number.
      </div>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}
          >
            <div
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                message.role === "user" ? "bg-slate-200 text-slate-600" : "bg-brand-600 text-white"
              }`}
            >
              {message.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
            </div>
            <div
              className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                message.role === "user"
                  ? "bg-brand-600 text-white"
                  : "border border-slate-200 bg-white text-slate-800"
              }`}
            >
              {message.content ||
                (isLoading && index === messages.length - 1 ? (
                  <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
                ) : (
                  ""
                ))}
            </div>
          </div>
        ))}
        {toolStatus && (
          <div className="flex items-center gap-2 pl-11 text-xs text-slate-500">
            <Wrench className="h-3.5 w-3.5 animate-pulse" />
            {toolStatus}
          </div>
        )}
        {!hasUserMessage && !compact && (
          <div className="flex flex-wrap gap-2 pl-11">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="rounded-full border border-brand-100 bg-brand-50 px-3 py-1.5 text-xs text-brand-700 transition hover:bg-brand-100"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        className="flex items-center gap-2 border-t border-slate-200 bg-white p-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about symptoms, stock, or appointments…"
          className="input"
          disabled={isLoading}
        />
        <button type="submit" className="btn-primary !px-3" disabled={isLoading || !input.trim()}>
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </button>
      </form>
    </div>
  );
}
