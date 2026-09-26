import pytest
from fakes import CitingLLM, HashEmbedder, OverlapReranker, make_pdf

from app.config import Settings
from app.db import connect, init_db
from app.ingest import PDFIngestor
from app.rag import RAGWorkflow
from app.retrieval import HybridRetriever

REPORT = [
    "Scope 1 emissions were 41,000 tonnes CO2e in fiscal 2024.",
    "Revenue grew 12 percent to 3.4 billion dollars, driven by services.",
    "The board approved a new water stewardship policy for all plants.",
]


@pytest.fixture()
def settings(tmp_path):
    return Settings(database_path=str(tmp_path / "rag.db"), retrieval_k=10, rerank_k=2)


@pytest.fixture()
def conn(settings):
    connection = connect(settings)
    init_db(connection, settings.embedding_dim)
    yield connection
    connection.close()


@pytest.fixture()
def ingestor(conn, settings):
    return PDFIngestor(conn, settings, embedder=HashEmbedder())


@pytest.fixture()
def retriever(conn, settings):
    return HybridRetriever(conn, settings, embedder=HashEmbedder(), reranker=OverlapReranker())


@pytest.fixture()
def report_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(make_pdf(REPORT))
    return path


@pytest.fixture()
def workflow(retriever, settings):
    return RAGWorkflow(retriever, settings, llm=CitingLLM())
