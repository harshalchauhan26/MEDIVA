import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertCircle,
  Bot,
  FileText,
  HeartPulse,
  Loader2,
  Send,
  Sparkles,
  User,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const starterPrompts = [
  "What are common symptoms of anemia?",
  "Explain diabetes mellitus in simple terms.",
  "What does the encyclopedia say about hypertension?",
];

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi, I’m MEDIVA. Ask a question and I’ll answer from the indexed medical reference documents.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const hasUserMessages = useMemo(
    () => messages.some((message) => message.role === "user"),
    [messages]
  );

  async function sendMessage(text = input) {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    setError("");
    setInput("");
    setIsLoading(true);
    setMessages((current) => [...current, { role: "user", content: trimmed }]);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "MEDIVA could not answer right now.");
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(err.message);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            "I couldn’t reach the medical knowledge service. Please check that the API is running and try again.",
          sources: [],
        },
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  return (
    <main className="min-h-screen bg-[#f7fbfa] text-ink">
      <section className="grid min-h-screen lg:grid-cols-[360px_1fr]">
        <aside className="flex flex-col justify-between border-b border-slate-200 bg-white px-6 py-7 lg:border-b-0 lg:border-r">
          <div>
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-md bg-mint text-white">
                <HeartPulse size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-semibold tracking-normal">MEDIVA</h1>
                <p className="text-sm text-slate-500">Medical RAG Assistant</p>
              </div>
            </div>

            <div className="mt-8 space-y-3">
              <div className="rounded-md border border-mint/20 bg-clinic p-4">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <Sparkles size={16} className="text-mint" />
                  Document-grounded answers
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Responses are generated from the local FAISS index built from the medical reference PDF.
                </p>
              </div>

              {!hasUserMessages && (
                <div className="space-y-2">
                  {starterPrompts.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      onClick={() => sendMessage(prompt)}
                      className="w-full rounded-md border border-slate-200 bg-white px-3 py-3 text-left text-sm text-slate-700 transition hover:border-mint hover:text-ink"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="mt-8 rounded-md border border-coral/20 bg-[#fff7f5] p-4 text-sm leading-6 text-slate-600">
            <div className="mb-2 flex items-center gap-2 font-semibold text-ink">
              <AlertCircle size={16} className="text-coral" />
              Clinical note
            </div>
            MEDIVA is for reference and learning. It is not a substitute for professional medical advice.
          </div>
        </aside>

        <section className="flex min-h-0 flex-col">
          <div className="border-b border-slate-200 bg-white/80 px-5 py-4 backdrop-blur">
            <div className="mx-auto flex max-w-4xl items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold">Medical document chat</h2>
                <p className="text-sm text-slate-500">Ask concise questions for the best retrieval.</p>
              </div>
              <div className="hidden items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-600 sm:flex">
                <FileText size={16} />
                FAISS index
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-6">
            <div className="mx-auto flex max-w-4xl flex-col gap-4">
              {messages.map((message, index) => (
                <MessageBubble key={`${message.role}-${index}`} message={message} />
              ))}
              {isLoading && (
                <div className="flex items-center gap-3 rounded-md border border-slate-200 bg-white p-4 shadow-sm">
                  <Loader2 size={18} className="animate-spin text-mint" />
                  <span className="text-sm text-slate-600">MEDIVA is checking the documents...</span>
                </div>
              )}
              {error && (
                <div className="rounded-md border border-coral/30 bg-[#fff7f5] p-4 text-sm text-coral">
                  {error}
                </div>
              )}
            </div>
          </div>

          <form onSubmit={handleSubmit} className="border-t border-slate-200 bg-white px-4 py-4">
            <div className="mx-auto flex max-w-4xl items-end gap-3">
              <label className="sr-only" htmlFor="prompt">
                Ask MEDIVA
              </label>
              <textarea
                ref={inputRef}
                id="prompt"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                  }
                }}
                rows={1}
                placeholder="Ask about symptoms, conditions, treatments, or medical terminology..."
                className="max-h-36 min-h-12 flex-1 resize-none rounded-md border border-slate-300 bg-white px-4 py-3 text-base outline-none transition placeholder:text-slate-400 focus:border-mint focus:ring-4 focus:ring-mint/10"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                aria-label="Send message"
                className="grid h-12 w-12 shrink-0 place-items-center rounded-md bg-ink text-white transition hover:bg-mint disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                <Send size={19} />
              </button>
            </div>
          </form>
        </section>
      </section>
    </main>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const Icon = isUser ? User : Bot;

  return (
    <article className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && <Avatar icon={Icon} tone="assistant" />}
      <div className={`max-w-[820px] ${isUser ? "order-first" : ""}`}>
        <div
          className={`rounded-md px-4 py-3 leading-7 shadow-sm ${
            isUser
              ? "bg-ink text-white"
              : "border border-slate-200 bg-white text-slate-700"
          }`}
        >
          {message.content}
        </div>
        {!isUser && message.sources?.length > 0 && (
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            {message.sources.map((source, index) => (
              <div
                key={`${source.source}-${source.page}-${index}`}
                className="rounded-md border border-slate-200 bg-white p-3 text-xs leading-5 text-slate-600"
              >
                <div className="mb-1 font-semibold text-ink">
                  {source.source || "Medical reference"}
                  {Number.isInteger(source.page) ? `, page ${source.page + 1}` : ""}
                </div>
                {source.preview}
              </div>
            ))}
          </div>
        )}
      </div>
      {isUser && <Avatar icon={Icon} tone="user" />}
    </article>
  );
}

function Avatar({ icon: Icon, tone }) {
  return (
    <div
      className={`grid h-9 w-9 shrink-0 place-items-center rounded-md ${
        tone === "user" ? "bg-ink text-white" : "bg-mint text-white"
      }`}
    >
      <Icon size={18} />
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
