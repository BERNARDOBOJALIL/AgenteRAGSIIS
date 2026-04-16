"""
api.py
API HTTP para el asistente universitario (RAG + Chroma Cloud + Groq).

Uso:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import generar_respuesta_rag, inicializar_vector_store

load_dotenv()


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    frontend_context: str | None = Field(default=None, max_length=4000)


class ChatResponse(BaseModel):
    response: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    collection_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        vector_store = inicializar_vector_store()
        app.state.vector_store = vector_store
        app.state.historiales = {}
    except Exception as e:
        raise RuntimeError(f"No se pudo iniciar la API: {e}") from e
    yield


app = FastAPI(
    title="API Asistente Universitario",
    description="API de chat para frontend usando RAG sobre Chroma Cloud.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store no inicializado")

    count = vector_store._collection.count()
    return HealthResponse(status="ok", collection_count=count)


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    vector_store = getattr(app.state, "vector_store", None)
    historiales = getattr(app.state, "historiales", None)

    if vector_store is None or historiales is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")

    historial = historiales.setdefault(body.session_id, [])

    try:
        respuesta = generar_respuesta_rag(
            vector_store,
            body.prompt.strip(),
            historial,
            frontend_context=(body.frontend_context or "").strip() or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando prompt: {e}") from e

    historial.append({"pregunta": body.prompt.strip(), "respuesta": respuesta})
    historiales[body.session_id] = historial[-10:]

    return ChatResponse(response=respuesta, session_id=body.session_id)


@app.delete("/chat/{session_id}")
def reset_session(session_id: str):
    historiales = getattr(app.state, "historiales", None)
    if historiales is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")

    historiales.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}
