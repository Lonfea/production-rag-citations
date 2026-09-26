import struct
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from app.config import Settings
from app.utils import fts5_query, reciprocal_rank_fusion

RetrievalMode = Literal["lexical", "semantic", "hybrid"]


class Embedder(Protocol):
    def encode(self, sentences, **kwargs): ...


class Reranker(Protocol):
    def predict(self, sentences, **kwargs): ...


@dataclass
class RetrievedChunk:
    id: int
    source: str
    page: int
    content: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _serialize_f32(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def load_embedder(settings: Settings) -> Embedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def load_reranker(settings: Settings) -> Reranker:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


class HybridRetriever:
    """Lexical (FTS5/BM25) and vector search fused with RRF, then cross-encoder reranking.

    Models are injectable so tests and benchmarks can run without downloading them.
    """

    def __init__(
        self,
        conn,
        settings: Settings,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.conn = conn
        self.settings = settings
        self._embedder = embedder
        self._reranker = reranker

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = load_embedder(self.settings)
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = load_reranker(self.settings)
        return self._reranker

    def _lexical(self, query: str) -> list[int]:
        parsed = fts5_query(query)
        if not parsed:
            return []
        rows = self.conn.execute(
            """
            SELECT chunk_id
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (parsed, self.settings.retrieval_k),
        ).fetchall()
        return [int(row["chunk_id"]) for row in rows]

    def _semantic(self, query: str) -> list[int]:
        embedding = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        rows = self.conn.execute(
            """
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (
                _serialize_f32(embedding.astype("float32").tolist()),
                self.settings.retrieval_k,
            ),
        ).fetchall()
        return [int(row["rowid"]) for row in rows]

    def candidates(self, query: str, mode: RetrievalMode = "hybrid") -> list[int]:
        """Chunk ids in first-stage rank order, before reranking."""
        if mode == "lexical":
            return self._lexical(query)
        if mode == "semantic":
            return self._semantic(query)
        fused = reciprocal_rank_fusion([self._lexical(query), self._semantic(query)])
        return [doc_id for doc_id, _ in fused[: self.settings.retrieval_k]]

    def search(
        self, query: str, mode: RetrievalMode = "hybrid", rerank: bool = True
    ) -> list[RetrievedChunk]:
        candidate_ids = self.candidates(query, mode)
        if not candidate_ids:
            return []

        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.conn.execute(
            f"SELECT id, source, page, content FROM chunks WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        ordered = [by_id[i] for i in candidate_ids if i in by_id]

        if rerank:
            scores = self.reranker.predict(
                [(query, row["content"]) for row in ordered],
                show_progress_bar=False,
            )
            ranked = sorted(
                zip(ordered, (float(score) for score in scores), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        else:
            # Without a reranker, keep first-stage order and expose rank as a score.
            ranked = [(row, 1.0 / rank) for rank, row in enumerate(ordered, start=1)]

        return [
            RetrievedChunk(
                id=int(row["id"]),
                source=row["source"],
                page=int(row["page"]),
                content=row["content"],
                score=score,
            )
            for row, score in ranked[: self.settings.rerank_k]
        ]
