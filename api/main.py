import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    page: int | None = None
    source: str | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


def _allowed_origins() -> list[str]:
    configured = os.getenv("FRONTEND_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


app = FastAPI(
    
    title="MEDIVA API",
    description="RAG API for the MEDIVA medical assistant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_rag_chain() -> Any:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    if not DB_FAISS_PATH.exists():
        raise RuntimeError(f"FAISS vectorstore was not found at {DB_FAISS_PATH}.")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL_NAME", DEFAULT_GROQ_MODEL),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.35")),
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "700")),
        api_key=groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are MEDIVA, a careful medical reference assistant. "
                "Answer using only the provided context. If the context does not contain "
                "the answer, say you do not know from the available medical documents. "
                "Do not diagnose, prescribe, or replace professional medical care.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, combine_docs_chain)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        response = get_rag_chain().invoke({"input": payload.message.strip()})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources: list[Source] = []
    for doc in response.get("context", []):
        metadata = doc.metadata or {}
        source_path = metadata.get("source")
        sources.append(
            Source(
                page=metadata.get("page"),
                source=Path(source_path).name if source_path else None,
                preview=doc.page_content[:240].strip(),
            )
        )

    return ChatResponse(answer=response.get("answer", ""), sources=sources)
