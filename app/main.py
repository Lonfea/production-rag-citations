import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import httpx
import openai
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db import connect, init_db
from app.ingest import PDFIngestor
from app.rag import RAGWorkflow
from app.retrieval import HybridRetriever, load_embedder

logger = logging.getLogger(__name__)

# What the Ollama (httpx) and OpenAI clients raise when their backend cannot be reached.
BACKEND_ERRORS = (httpx.TransportError, openai.APIConnectionError, openai.APITimeoutError)


@dataclass
class Services:
    settings: Settings
    ingestor: PDFIngestor
    workflow: RAGWorkflow


@lru_cache
def get_services() -> Services:
    """Build the index and models once, on first use rather than at import time."""
    settings = get_settings()
    conn = connect(settings)
    init_db(conn, settings.embedding_dim)
    embedder = load_embedder(settings)  # shared so the model is loaded once
    retriever = HybridRetriever(conn, settings, embedder=embedder)
    return Services(
        settings=settings,
        ingestor=PDFIngestor(conn, settings, embedder=embedder),
        workflow=RAGWorkflow(retriever, settings),
    )


ServicesDep = Annotated[Services, Depends(get_services)]

app = FastAPI(title="Production RAG with Citations", version="0.2.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)


class Source(BaseModel):
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": get_settings().model_provider}


@app.post("/ingest")
async def ingest(file: Annotated[UploadFile, File(...)], services: ServicesDep) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    limit = services.settings.max_upload_mb * 1024 * 1024
    payload = await file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {services.settings.max_upload_mb} MB upload limit.",
        )

    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    try:
        return services.ingestor.ingest(tmp_path, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, services: ServicesDep) -> AskResponse:
    try:
        result = services.workflow.ask(request.question)
    except BACKEND_ERRORS as exc:
        logger.exception("Language model backend unavailable")
        raise HTTPException(
            status_code=503, detail="The language model backend is unavailable."
        ) from exc
    sources = [
        Source(source=c["source"], page=c["page"], score=c["score"])
        for c in result.get("chunks", [])
    ]
    return AskResponse(
        answer=result["answer"],
        grounded=bool(result.get("grounded")),
        sources=sources,
    )
