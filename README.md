# MEDIVA

> **New:** the full healthcare platform (medicine inventory, appointment booking, and
> the MediVa tool-calling AI assistant) lives in [`webapp/`](webapp/README.md). It uses
> the RAG service below as-is — see [`webapp/ARCHITECTURE.md`](webapp/ARCHITECTURE.md).

MEDIVA is a medical Retrieval-Augmented Generation chatbot. The project has been migrated from a Streamlit-only app to a deployable React + Tailwind frontend and FastAPI backend.

The frontend is designed for Vercel. The backend is designed for Render and keeps the existing LangChain, Groq, HuggingFace embeddings, and FAISS vectorstore workflow.

## Project Structure

```text
MEDIVA/
├── api/
│   └── main.py                 # FastAPI RAG API for Render
├── frontend/
│   ├── src/
│   │   ├── main.jsx            # React chat UI
│   │   └── styles.css          # Tailwind entry
│   ├── package.json
│   └── tailwind.config.js
├── vectorstore/db_faiss/       # Existing FAISS index
├── data/                       # Source PDF documents
├── database.py                 # Rebuilds the FAISS vectorstore
├── render.yaml                 # Render deploy blueprint
├── vercel.json                 # Vercel frontend config
└── requirements.txt            # Backend dependencies
```

## Local Development

### Backend

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Set `GROQ_API_KEY` in `.env` before asking questions.

### Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## API

Health check:

```bash
GET /health
```

Chat:

```bash
POST /api/chat
Content-Type: application/json

{
  "message": "What are common symptoms of anemia?"
}
```

Response:

```json
{
  "answer": "Generated answer from the medical documents.",
  "sources": [
    {
      "page": 12,
      "source": "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf",
      "preview": "Retrieved source text..."
    }
  ]
}
```

## Deploy Backend To Render

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the repo.
3. Use `render.yaml`, or configure manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   - `GROQ_API_KEY`
   - `FRONTEND_ORIGINS=https://your-vercel-domain.vercel.app`
   - Optional: `GROQ_MODEL_NAME`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`

Render must include the committed `vectorstore/db_faiss` files, or you must rebuild them before deployment with:

```bash
python database.py
```

## Deploy Frontend To Vercel

1. Import the same GitHub repository into Vercel.
2. Vercel will use `vercel.json`:
   - Install command: `cd frontend && npm install`
   - Build command: `cd frontend && npm run build`
   - Output directory: `frontend/dist`
3. Add this environment variable:
   - `VITE_API_URL=https://your-render-service.onrender.com`
4. Redeploy after changing `VITE_API_URL`.

## Notes

- The old Streamlit app remains in `mainbot.py` as a legacy reference.
- The frontend calls only the FastAPI backend. It does not expose the Groq API key.
- MEDIVA is a reference and learning tool, not a replacement for professional medical advice.
