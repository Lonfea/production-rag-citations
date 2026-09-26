import httpx
import pytest
from fakes import make_pdf
from fastapi.testclient import TestClient

from app import main


@pytest.fixture()
def client(settings, ingestor, workflow):
    services = main.Services(settings=settings, ingestor=ingestor, workflow=workflow)
    main.app.dependency_overrides[main.get_services] = lambda: services
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def upload(client, name, payload):
    return client.post("/ingest", files={"file": (name, payload, "application/pdf")})


def test_upload_then_ask_returns_grounded_answer_with_sources(client):
    assert upload(client, "report.pdf", make_pdf(["Scope 1 emissions were 41,000 tonnes."])).status_code == 200
    response = client.post("/ask", json={"question": "What were Scope 1 emissions?"})
    body = response.json()
    assert response.status_code == 200
    assert body["grounded"] is True
    assert body["sources"][0] == {"source": "report.pdf", "page": 1, "score": body["sources"][0]["score"]}


def test_rejects_non_pdf_filename(client):
    assert upload(client, "notes.txt", b"hello").status_code == 400


def test_rejects_unreadable_pdf(client):
    assert upload(client, "broken.pdf", b"garbage").status_code == 422


def test_rejects_oversized_upload(client, settings):
    settings.max_upload_mb = 0
    assert upload(client, "big.pdf", make_pdf(["x"])).status_code == 413


def test_model_outage_returns_503(client, workflow):
    def unavailable(messages):
        raise httpx.ConnectError("connection refused")

    upload(client, "report.pdf", make_pdf(["Scope 1 emissions were 41,000 tonnes."]))
    workflow.llm.invoke = unavailable
    response = client.post("/ask", json={"question": "What were Scope 1 emissions?"})
    assert response.status_code == 503


def test_question_length_is_validated(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
