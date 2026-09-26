from fakes import CitingLLM

from app.rag import RAGWorkflow


def test_grounded_answer_cites_a_retrieved_page(ingestor, workflow, report_pdf):
    ingestor.ingest(report_pdf)
    result = workflow.ask("What were Scope 1 emissions?")
    assert result["grounded"] is True
    assert "[p.1]" in result["answer"]


def test_citation_to_unretrieved_page_is_replaced(ingestor, retriever, settings, report_pdf):
    ingestor.ingest(report_pdf)
    workflow = RAGWorkflow(retriever, settings, llm=CitingLLM(forced_page=99))
    result = workflow.ask("What were Scope 1 emissions?")
    assert result["grounded"] is False
    assert "grounding contract" in result["answer"]
    assert "[p.99]" not in result["answer"]


def test_empty_index_answers_without_calling_the_model(retriever, settings):
    llm = CitingLLM()
    result = RAGWorkflow(retriever, settings, llm=llm).ask("What were emissions?")
    assert result["grounded"] is False
    assert "do not contain enough evidence" in result["answer"]
    assert llm.prompts == []


def test_evidence_is_labelled_with_page_and_source(ingestor, retriever, settings, report_pdf):
    ingestor.ingest(report_pdf)
    llm = CitingLLM()
    RAGWorkflow(retriever, settings, llm=llm).ask("revenue growth")
    assert "[p.2 | report.pdf]" in llm.prompts[0]
