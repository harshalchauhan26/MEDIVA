import ChatPanel from "@/components/ChatPanel";

export const metadata = { title: "MediVa AI — MEDIVA" };

export default function ChatPage() {
  return (
    <div className="card mx-auto flex h-[calc(100vh-14rem)] max-w-3xl flex-col overflow-hidden">
      <div className="border-b border-slate-200 px-6 py-4">
        <h1 className="font-semibold">MediVa — AI Health Assistant</h1>
        <p className="text-sm text-slate-500">
          Grounded in indexed medical reference documents, with live access to inventory and
          appointment availability.
        </p>
      </div>
      <div className="min-h-0 flex-1">
        <ChatPanel />
      </div>
    </div>
  );
}
