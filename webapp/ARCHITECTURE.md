# MEDIVA Platform Architecture

## System Overview

```mermaid
flowchart TB
    subgraph Browser
        UI[Next.js React UI<br/>Tailwind + React Query]
        Widget[Floating MediVa Widget<br/>+ /chat dashboard]
    end

    subgraph NextJS["Next.js App (webapp/) — Vercel/Node"]
        REST[REST Route Handlers<br/>/api/medicines /api/doctors /api/appointments /api/auth]
        Agent[MediVa Agent<br/>/api/chat — Groq tool calling]
    end

    subgraph Data["PostgreSQL + pgvector"]
        SQL[(Users, Doctors, Appointments,<br/>Medicines, Reservations)]
        PGV[(MedicalDocuments<br/>vector(384) — future RAG home)]
    end

    subgraph RAG["EXISTING RAG service (api/main.py) — UNCHANGED"]
        FastAPI[FastAPI /api/chat]
        FAISS[(FAISS index<br/>MiniLM-L6-v2 embeddings)]
        Groq1[Groq LLM<br/>llama-3.1-8b-instant]
    end

    UI -->|React Query fetch| REST
    Widget -->|NDJSON stream| Agent
    REST --> SQL
    Agent -->|"SQL tools: stock, slots, booking"| SQL
    Agent -->|"search_medical_knowledge tool"| FastAPI
    FastAPI --> FAISS --> Groq1
    Agent -->|tool-use loop| Groq2[Groq LLM<br/>llama-3.3-70b-versatile]
```

## How the agent decides: RAG vs SQL (tool calling)

The MediVa agent in `src/app/api/chat/route.ts` does **not** hardcode intent routing.
It hands the LLM (Groq `llama-3.3-70b-versatile`) four tool definitions and lets the
model pick per turn, guided by the system prompt in `src/lib/agent.ts`:

| User intent | Tool the model calls | Backing store |
|---|---|---|
| "What are the symptoms of anemia?" | `search_medical_knowledge` | **Existing FastAPI RAG** (FAISS + Gale Encyclopedia) — proxied over HTTP, untouched |
| "Is paracetamol in stock?" | `check_medicine_stock` | PostgreSQL `Medicine` (live Prisma query) |
| "Open slot with a cardiologist tomorrow?" | `find_available_slots` | PostgreSQL `Doctor` + `AvailabilityBlock` − booked `Appointment` rows |
| "Book it for 10:30, I have chest pain" | `book_appointment` | PostgreSQL `Appointment` (same `bookAppointment()` used by the REST route) |

The loop (max 5 rounds):

1. Send conversation + tool schemas to Groq with `tool_choice: "auto"`.
2. If the model returns `tool_calls`, execute each server-side (Prisma for SQL tools;
   `fetch` to `RAG_API_URL` for knowledge), append `role:"tool"` results, repeat.
3. When the model returns plain content, stream it to the browser as NDJSON
   (`{type:"tool"}` progress events + `{type:"token"}` text chunks).

Key properties:

- **The RAG pipeline is a black box dependency.** The agent treats your FastAPI
  service exactly like a tool — nothing in `api/main.py`, `database.py`, or the FAISS
  index changed.
- **Authorization lives in the tools, not the model.** `book_appointment` re-checks the
  signed session cookie server-side; an unauthenticated user gets a polite "please log
  in" regardless of what the LLM tries.
- **Booking can never diverge** between chat and the UI: both call the same
  `src/lib/booking.ts`, which validates the slot against availability minus existing
  appointments and relies on a DB unique constraint `(doctorId, startsAt)` to kill
  race conditions.

## Data flow examples

**Reserve for pickup:** UI → `POST /api/medicines/:id/reserve` → Prisma transaction
(`updateMany` with `quantity >= n` guard, then `Reservation.create`) → mock email →
React Query invalidates `["medicines"]` so stock re-renders everywhere, including the
admin dashboard's low-stock alerts.

**Booking via chat:** widget → `/api/chat` → model calls `find_available_slots` →
user confirms → model calls `book_appointment` → `bookAppointment()` → `PENDING` row +
mock email/SMS → doctor sees it on `/doctor/dashboard` (polling every 20s) and
confirms → `PATCH /api/appointments/:id` → mock status email.

## Roles & access

| Capability | Patient | Doctor | Pharmacist | Admin |
|---|---|---|---|---|
| Browse stock / doctors / chat | ✅ | ✅ | ✅ | ✅ |
| Reserve medicine, book appointment | ✅ | — | — | — |
| Confirm/cancel/complete own appointments | cancel own | ✅ | — | — |
| Inventory CRUD | — | — | ✅ | ✅ |

Sessions are HMAC-signed httpOnly cookies (`src/lib/auth.ts`); every mutating route
re-validates the role server-side.

## pgvector note

`MedicalDocument` (vector(384), matching all-MiniLM-L6-v2) is provisioned so document
chunks can later be ingested into Postgres and the knowledge tool repointed — without
touching the current FAISS service. Until then, FAISS remains the live RAG store.
