import hashlib
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings
from app.retrieval import Embedder, _serialize_f32, load_embedder


def _chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


class PDFIngestor:
    def __init__(self, conn, settings: Settings, embedder: Embedder | None = None):
        self.conn = conn
        self.settings = settings
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = load_embedder(self.settings)
        return self._embedder

    @staticmethod
    def _extract_pages(pdf_path: Path) -> list[tuple[int, int, str]]:
        try:
            reader = PdfReader(str(pdf_path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, ValueError, KeyError) as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc
        return [
            (page_number, chunk_index, chunk)
            for page_number, text in enumerate(pages, start=1)
            for chunk_index, chunk in enumerate(_chunk_text(text))
        ]

    def ingest(self, pdf_path: Path, source_name: str | None = None) -> dict:
        source = source_name or pdf_path.name
        payload = pdf_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()

        existing = self.conn.execute(
            "SELECT id, sha256 FROM documents WHERE source = ?", (source,)
        ).fetchone()
        if existing and existing["sha256"] == digest:
            count = self.conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?", (existing["id"],)
            ).fetchone()["n"]
            return {"source": source, "chunks": count, "status": "unchanged"}

        # Parse and embed before touching the index, then swap versions in one
        # transaction, so a bad upload or a model failure cannot lose the old version.
        rows = self._extract_pages(pdf_path)
        if not rows:
            raise ValueError("No extractable text found. Scanned PDFs require OCR.")
        embeddings = self.embedder.encode(
            [r[2] for r in rows],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        with self.conn:
            if existing:
                self._delete_document(existing["id"])
            cur = self.conn.execute(
                "INSERT INTO documents(source, sha256) VALUES (?, ?)", (source, digest)
            )
            document_id = cur.lastrowid

            for (page, chunk_index, content), embedding in zip(rows, embeddings, strict=True):
                cur = self.conn.execute(
                    """
                    INSERT INTO chunks(document_id, source, page, chunk_index, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, source, page, chunk_index, content),
                )
                chunk_id = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, content) VALUES (?, ?)",
                    (chunk_id, content),
                )
                self.conn.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, _serialize_f32(embedding.astype("float32").tolist())),
                )

        return {"source": source, "chunks": len(rows), "status": "indexed"}

    def _delete_document(self, document_id: int) -> None:
        ids = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            )
        ]
        for chunk_id in ids:
            self.conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
            self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        self.conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
