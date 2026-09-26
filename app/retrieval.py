import struct
from dataclasses import asdict, dataclass

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import Settings
from app.utils import fts5_query, reciprocal_rank_fusion


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


class HybridRetriever:
    def __init__(self, conn, settings: Settings):
        self.conn = conn
        self.settings = settings
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.reranker = CrossEncoder(settings.rerank_model)

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

    def search(self, query: str) -> list[RetrievedChunk]:
        lexical = self._lexical(query)
        semantic = self._semantic(query)
        fused = reciprocal_rank_fusion([lexical, semantic])
        candidate_ids = [doc_id for doc_id, _ in fused[: self.settings.retrieval_k]]
        if not candidate_ids:
            return []

        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.conn.execute(
            f"SELECT id, source, page, content FROM chunks WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        ordered = [by_id[i] for i in candidate_ids if i in by_id]

        scores = self.reranker.predict(
            [(query, row["content"]) for row in ordered],
            show_progress_bar=False,
        )

        reranked = sorted(
            zip(ordered, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[: self.settings.rerank_k]

        return [
            RetrievedChunk(
                id=int(row["id"]),
                source=row["source"],
                page=int(row["page"]),
                content=row["content"],
                score=float(score),
            )
            for row, score in reranked
        ]
