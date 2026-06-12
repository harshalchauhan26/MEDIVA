# MediVa — Production Deployment & Real-User Evaluation Guide

This guide takes MediVa from local dev to a live URL real users can sign up to and
evaluate. Three pieces are deployed:

| Piece | What it is | Where it goes |
|---|---|---|
| **Database** | PostgreSQL + pgvector | **Neon** (managed, free tier) |
| **RAG service** | FastAPI + FAISS (`../api/main.py`) | **Render** (free tier) |
| **Web app** | Next.js 14 (`webapp/`) | **Vercel** (free tier) |

```
            ┌─────────────┐     server-side fetch      ┌──────────────┐
 Browser ──▶│  Vercel      │ ─────────────────────────▶│  Render       │
            │  (Next.js)   │   RAG_API_URL /api/chat    │  (FastAPI)    │
            └──────┬───────┘                            └──────────────┘
                   │ DATABASE_URL (Prisma)
                   ▼
            ┌─────────────┐
            │  Neon        │  Postgres + pgvector
            └─────────────┘
```

> Order matters: **Database → RAG service → Web app**, because the web app needs
> both URLs/keys before its first build.

---

## 0. Prerequisites

- A GitHub repo with this code pushed.
- A [Groq API key](https://console.groq.com) (free).
- Accounts on [Neon](https://neon.tech), [Render](https://render.com), [Vercel](https://vercel.com) (all free).
- Optional: Google Cloud project for "Sign in with Google", Meta WhatsApp Cloud API for real confirmations. Both have working fallbacks (email/password login, mock WhatsApp), so you can launch without them.

---

## 1. Database — Neon (Postgres + pgvector)

1. Create a Neon project. Region close to your users (e.g. AWS `ap-south-1` for India).
2. In the Neon SQL editor, enable pgvector once:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Copy the **pooled** connection string (the host contains `-pooler`). Append
   `?sslmode=require`. This is your `DATABASE_URL`:
   ```
   postgresql://USER:PASSWORD@ep-xxxx-pooler.ap-south-1.aws.neon.tech/neondb?sslmode=require
   ```

You'll push the schema and seed data in step 4 (after the web app has the URL),
or do it now locally:

```bash
cd webapp
DATABASE_URL="<neon-url>" npm run db:push     # create tables
DATABASE_URL="<neon-url>" npm run db:seed     # demo doctors, medicines, accounts
```

> On Windows PowerShell: `$env:DATABASE_URL="<neon-url>"; npm run db:push`.

---

## 2. RAG service — Render (FastAPI + FAISS)

The blueprint already exists at the repo root: [`../render.yaml`](../render.yaml).

1. In Render: **New ▸ Blueprint**, point it at your GitHub repo. Render reads
   `render.yaml` and provisions the `mediva-api` web service.
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
2. Set environment variables on the service:
   - `GROQ_API_KEY` — your Groq key.
   - `FRONTEND_ORIGINS` — your Vercel domain once known (e.g.
     `https://mediva.vercel.app`). The web app calls the RAG service
     **server-side**, so CORS is only needed if you also use the legacy browser
     frontend; setting it is harmless.
3. The committed FAISS index in `vectorstore/db_faiss/` ships with the repo. If it's
   missing, rebuild before/at deploy: `python database.py`.
4. After deploy, confirm health:
   ```
   GET https://<your-service>.onrender.com/health   ->   {"status":"ok"}
   ```
   Note this base URL — it's your `RAG_API_URL`.

> Free Render instances sleep after inactivity; the first chat after idle takes a
> few seconds to wake. Fine for evaluation.

---

## 3. Web app — Vercel (Next.js)

1. In Vercel: **Add New ▸ Project**, import the repo.
2. **Set Root Directory to `webapp`.** Vercel auto-detects Next.js.
   - Build command (default): `npm run build` — `prisma generate` runs automatically
     via the `postinstall` hook.
3. Add Environment Variables (Production + Preview):

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | Neon pooled URL from step 1 |
   | `GROQ_API_KEY` | your Groq key |
   | `AGENT_MODEL` | `llama-3.3-70b-versatile` |
   | `RAG_API_URL` | Render URL from step 2 |
   | `AUTH_SECRET` | long random string (see below) |
   | `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | optional |
   | `WHATSAPP_TOKEN` / `WHATSAPP_PHONE_NUMBER_ID` | optional |

   Generate `AUTH_SECRET`:
   ```bash
   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
   ```
4. Deploy. Then run the one-time DB setup against Neon if you didn't in step 1:
   ```bash
   cd webapp
   $env:DATABASE_URL="<neon-url>"; npm run db:push; npm run db:seed
   ```
5. Open `https://<your-app>.vercel.app` — you should land on the marketing page with
   **Sign up free / Log in**.

---

## 4. Optional integrations

### Google sign-in
1. Google Cloud Console ▸ APIs & Services ▸ Credentials ▸ **OAuth client ID** (Web).
2. Authorized redirect URI: `https://<your-app>.vercel.app/api/auth/google/callback`.
3. Put the client ID/secret in Vercel env vars and redeploy. The "Continue with
   Google" buttons activate automatically.

### WhatsApp confirmations
Set `WHATSAPP_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID` from the Meta WhatsApp Cloud API.
Without them, the app logs the message server-side and produces a `wa.me` link — good
enough for evaluation.

---

## 5. Smoke test the live deployment

Run through this once after deploying:

- [ ] `/health` on Render returns `{"status":"ok"}`.
- [ ] Marketing page loads; **Sign up free** works and creates a Patient account.
- [ ] After sign-up you're sent to onboarding (if no phone) then `/doctors`.
- [ ] Log out, log back in with the same email/password.
- [ ] Demo accounts still work: `admin@mediva.dev` / `mediva123`.
- [ ] `/pharmacy` lists seeded medicines; **Reserve for Pickup** decrements stock.
- [ ] `/doctors` shows live slots; booking creates a PENDING appointment.
- [ ] MediVa chat (bottom-right) answers a medical question (RAG) **and** a stock
      question (DB) — confirms the Render link works.
- [ ] The **Feedback** button (bottom-left) submits; as `admin@mediva.dev` you can
      read it at `/admin/feedback`.

---

## 6. Running a real-user evaluation

The app is now set up for it:

- **Open self-sign-up** at `/signup` — every evaluator gets their own Patient account
  (no shared demo logins needed). Staff roles (Doctor/Pharmacist/Admin) remain seed-only.
- **In-app feedback widget** on every page (1–5 stars + comment, optional email).
  Submissions are stored in the `Feedback` table.
- **Admin review** at `/admin/feedback` (log in as `admin@mediva.dev`) shows response
  count, average rating, and every comment with the page it came from.

Suggested flow:
1. Share the Vercel URL + a one-line task list ("sign up, book an appointment, reserve
   a medicine, ask MediVa a question, then leave feedback").
2. Watch ratings/comments accumulate in `/admin/feedback`.
3. Export raw rows from Neon if you want to analyze:
   `SELECT rating, message, page, "createdAt" FROM "Feedback" ORDER BY "createdAt" DESC;`

---

## 7. Updating after launch

- **Code**: push to GitHub → Vercel/Render auto-deploy.
- **Schema change** (e.g. the new `Feedback` table): re-run `npm run db:push` against
  `DATABASE_URL`. This was added in this release, so run it once before evaluating.
- **Secrets**: rotate in the Vercel/Render dashboards; never commit `.env`.
