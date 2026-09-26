import re
from collections.abc import Iterable


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked document ids using Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


_CITATION_RE = re.compile(r"\[p\.(\d+)\]")


def extract_cited_pages(answer: str) -> set[int]:
    return {int(page) for page in _CITATION_RE.findall(answer)}


def citations_are_grounded(answer: str, retrieved_pages: Iterable[int]) -> bool:
    cited = extract_cited_pages(answer)
    allowed = set(retrieved_pages)
    return bool(cited) and cited.issubset(allowed)


def fts5_query(text: str) -> str:
    """Create a conservative FTS5 OR query from user input."""
    terms = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return " OR ".join(f'"{term}"' for term in terms[:24])
