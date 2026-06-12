# MEDIVA Web Platform

Full-stack healthcare app: medicine inventory, doctor appointment booking, and the
MediVa 24/7 AI assistant (tool-calling agent grounded in the existing RAG service).

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the system diagram and agent data flow,
and [DEPLOYMENT.md](./DEPLOYMENT.md) for the full production deploy + real-user
evaluation guide (Neon + Render + Vercel).

## Stack

- Next.js 14 (App Router) + TypeScript + Tailwind CSS
- TanStack React Query for live inventory/appointment caching
- PostgreSQL + pgvector via Prisma
- Groq (`llama-3.3-70b-versatile`) tool-calling agent
- Google OAuth sign-in (+ email/password demo accounts)
- WhatsApp confirmations (Meta WhatsApp Cloud API, with a mock/`wa.me` fallback)
- Online buy links per medicine (Tata 1mg, PharmEasy, Netmeds, Apollo); all prices in ₹ (INR)
- Existing FastAPI + FAISS RAG service (`../api/main.py`) consumed as a tool — **unchanged**

## Quick start

```bash
# 1. Database (PostgreSQL + pgvector)
docker compose up -d

# 2. Environment
copy .env.example .env        # then fill in GROQ_API_KEY
#   Optional: GOOGLE_CLIENT_ID/SECRET for Google sign-in,
#   WHATSAPP_TOKEN/PHONE_NUMBER_ID for real WhatsApp delivery.
#   Without these, email/password login and mock WhatsApp (logged + wa.me link) work.

# 3. Install, create schema, seed demo data
npm install
npm run db:push
npm run db:seed

# 4. Start the EXISTING RAG service from the repo root (separate terminal)
cd ..
uvicorn api.main:app --reload --port 8000

# 5. Start the web app
npm run dev                   # http://localhost:3001 (3000 is often taken on this machine)
```

## Accounts

Real users **self-register** at `/signup` (email + password, or Google) and become
**Patients**. Staff roles are seed-only or admin-promoted.

### Demo accounts (password: `mediva123`)

| Role | Email | Landing page |
|---|---|---|
| Patient | patient@mediva.dev | /doctors |
| Doctor | dr.rao@mediva.dev (also dr.mehta, dr.iyer) | /doctor/dashboard |
| Pharmacist | pharma@mediva.dev | /admin/inventory |
| Admin | admin@mediva.dev | /admin/inventory |

## Pages

- `/` — dashboard with live platform stats
- `/chat` — full-screen MediVa; a floating widget is also on every other page
- `/pharmacy` — patient stock search + Reserve for Pickup
- `/admin/inventory` — pharmacist CRUD, low-stock (<10) and expiry (<60 days) alerts
- `/doctors` — specialty/rating filters, 14-day calendar of live open slots, booking form
- `/doctor/dashboard` — schedule, patient list, status toggle, confirm/cancel/complete
- `/signup` — public patient self-registration (email/password + Google)
- `/admin/feedback` — admin-only view of evaluation feedback (rating, comments)

A floating **Feedback** widget (bottom-left, all pages) lets evaluators submit a 1–5
rating + comment; submissions land in the `Feedback` table and surface at `/admin/feedback`.

## API

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /api/auth/signup` | public | Patient self-registration |
| `POST /api/auth/login` / `logout` / `GET me` | — | Cookie sessions |
| `POST /api/feedback` | public | Submit evaluation feedback |
| `GET /api/feedback` | admin | Read collected feedback |
| `GET /api/medicines?q=` | public | Search inventory |
| `POST /api/medicines`, `PATCH/DELETE /api/medicines/:id` | pharmacist/admin | Inventory CRUD |
| `POST /api/medicines/:id/reserve` | patient | Atomic stock decrement + reservation |
| `GET /api/doctors?specialty=&minRating=` | public | Doctor directory |
| `GET /api/doctors/:id/slots?date=YYYY-MM-DD` | public | Live open slots |
| `GET/PATCH /api/doctors/me` | doctor | Dashboard data, status toggle |
| `GET/POST /api/appointments`, `PATCH /api/appointments/:id` | patient/doctor | Booking lifecycle |
| `POST /api/chat` | optional | MediVa agent (NDJSON stream) |

Notifications (booking/reservation/status emails + SMS) are mocked: they print to the
Next.js server console via `src/lib/notifications.ts`.

## Deploy

- **Web app:** Vercel (set `DATABASE_URL`, `GROQ_API_KEY`, `RAG_API_URL`, `AUTH_SECRET`).
  Use a managed Postgres with pgvector (Neon/Supabase) and run `npm run db:push && npm run db:seed` once.
- **RAG service:** stays on Render exactly as before (`../render.yaml`).
