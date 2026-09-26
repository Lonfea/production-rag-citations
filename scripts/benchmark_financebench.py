"""Retrieval benchmark on FinanceBench (open-source subset).

FinanceBench (Islam et al., 2023) has 150 questions over 84 public SEC
filings. Each question is labelled with the document and page that hold
the evidence, which makes page-level retrieval measurable.

All 84 filings go into one shared index (about 13,000 pages), so the
retriever must find the right company and year as well as the right page.
A question counts as a hit at k if any of the top-k chunks comes from one
of its evidence pages.

Examples:

    # lexical only: runs without downloading any model
    python scripts/benchmark_financebench.py --modes lexical

    # full ablation: downloads the embedding and reranking models
    python scripts/benchmark_financebench.py --modes lexical,semantic,hybrid,hybrid+rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.db import connect, init_db
from app.ingest import PDFIngestor
from app.retrieval import HybridRetriever, load_embedder

REPO = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS = "financebench_open_source.jsonl"
MODES = {
    "lexical": ("lexical", False),
    "semantic": ("semantic", False),
    "hybrid": ("hybrid", False),
    "lexical+rerank": ("lexical", True),
    "hybrid+rerank": ("hybrid", True),
}
CUTOFFS = (1, 3, 5, 10)


class ZeroEmbedder:
    """Placeholder vectors for lexical-only runs, so no model is downloaded."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, sentences, **kwargs):
        return np.zeros((len(sentences), self.dim), dtype="float32")


def fetch(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
    return target


def load_questions(data_dir: Path) -> list[dict]:
    path = fetch(f"{REPO}/data/{QUESTIONS}", data_dir / QUESTIONS)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gold_pages(question: dict) -> set[tuple[str, int]]:
    # FinanceBench page numbers are 0-indexed; the index stores 1-indexed pages.
    return {(f"{e['doc_name']}.pdf", int(e["evidence_page_num"]) + 1) for e in question["evidence"]}


def build_index(data_dir: Path, questions: list[dict], settings: Settings, embedder) -> dict:
    documents = sorted({q["doc_name"] for q in questions})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda name: fetch(f"{REPO}/pdfs/{name}.pdf", data_dir / "pdfs" / f"{name}.pdf"),
                documents,
            )
        )
    conn = connect(settings)
    init_db(conn, settings.embedding_dim)
    ingestor = PDFIngestor(conn, settings, embedder=embedder)
    started = time.perf_counter()
    for name in documents:
        ingestor.ingest(data_dir / "pdfs" / f"{name}.pdf")
    stats = conn.execute(
        "SELECT COUNT(DISTINCT source) AS documents, COUNT(DISTINCT source || page) AS pages, "
        "COUNT(*) AS chunks FROM chunks"
    ).fetchone()
    return {
        "conn": conn,
        "documents": stats["documents"],
        "pages": stats["pages"],
        "chunks": stats["chunks"],
        "ingest_seconds": round(time.perf_counter() - started, 1),
    }


def evaluate(retriever: HybridRetriever, questions: list[dict], mode: str, rerank: bool) -> dict:
    hits = defaultdict(int)
    doc_hits = 0
    reciprocal_ranks = []
    by_type: dict[str, list[bool]] = defaultdict(list)
    latencies = []
    for question in questions:
        gold = gold_pages(question)
        gold_docs = {source for source, _ in gold}
        started = time.perf_counter()
        results = retriever.search(question["question"], mode=mode, rerank=rerank)
        latencies.append(time.perf_counter() - started)
        ranked = [(chunk.source, chunk.page) for chunk in results]
        first_hit = next((rank for rank, key in enumerate(ranked, start=1) if key in gold), None)
        for cutoff in CUTOFFS:
            hits[cutoff] += int(first_hit is not None and first_hit <= cutoff)
        doc_hits += int(any(source in gold_docs for source, _ in ranked[:5]))
        reciprocal_ranks.append(1 / first_hit if first_hit else 0.0)
        by_type[question["question_type"]].append(first_hit is not None and first_hit <= 5)

    total = len(questions)
    return {
        **{f"page_hit@{cutoff}": round(hits[cutoff] / total, 3) for cutoff in CUTOFFS},
        "doc_hit@5": round(doc_hits / total, 3),
        "mrr@10": round(sum(reciprocal_ranks) / total, 3),
        "page_hit@5_by_question_type": {
            kind: round(sum(values) / len(values), 3) for kind, values in sorted(by_type.items())
        },
        "median_latency_ms": round(float(np.median(latencies)) * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "data")
    parser.add_argument("--modes", default="lexical", help=f"comma-separated: {','.join(MODES)}")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",")]
    unknown = set(modes) - set(MODES)
    if unknown:
        parser.error(f"unknown modes: {sorted(unknown)}")
    needs_embeddings = any(MODES[mode][0] != "lexical" for mode in modes)

    settings = Settings(
        database_path=str(args.data_dir / ("index.db" if needs_embeddings else "index-lexical.db")),
        retrieval_k=20,
        rerank_k=max(CUTOFFS),
    )
    embedder = load_embedder(settings) if needs_embeddings else ZeroEmbedder(settings.embedding_dim)
    questions = load_questions(args.data_dir)
    index = build_index(args.data_dir, questions, settings, embedder)
    retriever = HybridRetriever(index.pop("conn"), settings, embedder=embedder)

    report = {
        "dataset": "FinanceBench open-source subset (patronus-ai/financebench)",
        "questions": len(questions),
        "index": index,
        "chunking": "1,200 characters with 200 overlap, within pages",
        "first_stage_candidates": settings.retrieval_k,
        "embedding_model": settings.embedding_model if needs_embeddings else None,
        "rerank_model": settings.rerank_model if any(MODES[m][1] for m in modes) else None,
        "results": {mode: evaluate(retriever, questions, *MODES[mode]) for mode in modes},
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
