from app.utils import citations_are_grounded, extract_cited_pages, reciprocal_rank_fusion


def test_rrf_rewards_consensus():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]])
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] in {1, 2}
    assert ids.index(1) < ids.index(3)
    assert ids.index(2) < ids.index(4)


def test_extract_cited_pages_deduplicates():
    assert extract_cited_pages("Claim [p.3]. More [p.3] and [p.8].") == {3, 8}


def test_grounding_accepts_only_retrieved_pages():
    assert citations_are_grounded("Supported [p.2] and [p.5].", [2, 5, 9])


def test_grounding_rejects_unknown_page():
    assert not citations_are_grounded("Unsupported [p.99].", [2, 5, 9])


def test_grounding_requires_a_citation():
    assert not citations_are_grounded("No citation here.", [2, 5, 9])
