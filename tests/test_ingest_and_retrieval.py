import pytest
from fakes import HashEmbedder, make_pdf

from app.ingest import PDFIngestor


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]


def test_ingest_indexes_every_page_in_all_three_tables(ingestor, conn, report_pdf):
    result = ingestor.ingest(report_pdf)
    assert result == {"source": "report.pdf", "chunks": 3, "status": "indexed"}
    assert count(conn, "chunks") == count(conn, "chunks_fts") == count(conn, "chunks_vec") == 3


def test_reingesting_identical_file_is_a_no_op(ingestor, report_pdf):
    ingestor.ingest(report_pdf)
    assert ingestor.ingest(report_pdf)["status"] == "unchanged"


def test_changed_file_replaces_previous_version(ingestor, conn, report_pdf):
    ingestor.ingest(report_pdf)
    report_pdf.write_bytes(make_pdf(["Only one page now."]))
    assert ingestor.ingest(report_pdf)["status"] == "indexed"
    assert count(conn, "documents") == 1
    assert count(conn, "chunks") == count(conn, "chunks_fts") == count(conn, "chunks_vec") == 1


def test_unreadable_upload_keeps_existing_version(ingestor, conn, report_pdf):
    ingestor.ingest(report_pdf)
    report_pdf.write_bytes(b"not a pdf at all")
    with pytest.raises(ValueError, match="Could not read PDF"):
        ingestor.ingest(report_pdf)
    assert count(conn, "chunks") == 3


def test_embedding_failure_keeps_existing_version(conn, settings, report_pdf):
    PDFIngestor(conn, settings, embedder=HashEmbedder()).ingest(report_pdf)

    class BrokenEmbedder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("model crashed")

    report_pdf.write_bytes(make_pdf(["A newer version."]))
    with pytest.raises(RuntimeError):
        PDFIngestor(conn, settings, embedder=BrokenEmbedder()).ingest(report_pdf)
    assert count(conn, "chunks") == 3


def test_text_free_pdf_is_rejected(ingestor, tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf([""]))
    with pytest.raises(ValueError, match="OCR"):
        ingestor.ingest(path)


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_every_mode_finds_the_page_with_the_answer(ingestor, retriever, report_pdf, mode):
    ingestor.ingest(report_pdf)
    results = retriever.search("How much did revenue grow?", mode=mode)
    assert results[0].page == 2


def test_reranked_results_are_sorted_by_reranker_score_and_capped(ingestor, retriever, report_pdf):
    ingestor.ingest(report_pdf)
    reranked = retriever.search("water stewardship policy approved by the board", rerank=True)
    assert reranked[0].page == 3
    assert [chunk.score for chunk in reranked] == sorted((c.score for c in reranked), reverse=True)
    assert len(reranked) == 2  # rerank_k caps the context


def test_query_without_searchable_terms_returns_nothing(ingestor, retriever, report_pdf):
    ingestor.ingest(report_pdf)
    assert retriever.search("?!", mode="lexical") == []
