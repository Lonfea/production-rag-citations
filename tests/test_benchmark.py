import benchmark_financebench as benchmark
from fakes import make_pdf


def test_gold_pages_convert_financebench_zero_indexed_pages():
    question = {"evidence": [{"doc_name": "3M_2018_10K", "evidence_page_num": 59}]}
    assert benchmark.gold_pages(question) == {("3M_2018_10K.pdf", 60)}


def test_evaluate_scores_hits_by_rank(ingestor, retriever, tmp_path):
    path = tmp_path / "ACME_2024_10K.pdf"
    path.write_bytes(make_pdf(["Cover page", "Capital expenditure was 1,577 million dollars"]))
    ingestor.ingest(path)
    questions = [
        {
            "question": "What was capital expenditure?",
            "question_type": "metrics-generated",
            "evidence": [{"doc_name": "ACME_2024_10K", "evidence_page_num": 1}],
        },
        {
            "question": "What was capital expenditure?",
            "question_type": "novel-generated",
            "evidence": [{"doc_name": "OTHER_2024_10K", "evidence_page_num": 0}],
        },
    ]
    result = benchmark.evaluate(retriever, questions, mode="lexical", rerank=False)
    assert result["page_hit@1"] == 0.5
    assert result["doc_hit@5"] == 0.5
    assert result["page_hit@5_by_question_type"] == {"metrics-generated": 1.0, "novel-generated": 0.0}
