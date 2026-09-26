from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import connect, init_db
from app.ingest import PDFIngestor
from app.rag import RAGWorkflow
from app.retrieval import HybridRetriever

settings = get_settings()
conn = connect(settings)
init_db(conn, settings.embedding_dim)

ingestor = PDFIngestor(conn, settings)
retriever = HybridRetriever(conn, settings)
workflow = RAGWorkflow(retriever, settings)

app = FastAPI(title="Production RAG with Citations", version="0.1.0")


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": settings.model_provider}


@app.post("/ingest")
async def ingest(file: Annotated[UploadFile, File(...)]) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    payload = await file.read()
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    try:
        return ingestor.ingest(tmp_path, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = workflow.ask(request.question)
    sources = [
        Source(source=c["source"], page=c["page"], score=c["score"])
        for c in result.get("chunks", [])
    ]
    return AskResponse(
        answer=result["answer"],
        grounded=bool(result.get("grounded")),
        sources=sources,
    )
