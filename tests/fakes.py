"""Small stand-ins for the embedding model, reranker and LLM.

They are deterministic and need no downloads, so tests exercise the real
SQLite index, fusion, reranking order and grounding logic.
"""

import re
import zlib

import numpy as np

DIM = 384


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HashEmbedder:
    """Bag-of-words vectors via the hashing trick; similar text gets similar vectors."""

    def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
        vectors = np.zeros((len(sentences), DIM), dtype="float32")
        for row, sentence in enumerate(sentences):
            for token in _tokens(sentence):
                vectors[row, zlib.crc32(token.encode()) % DIM] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


class OverlapReranker:
    """Scores a (query, passage) pair by the number of shared distinct tokens."""

    def predict(self, pairs, show_progress_bar=False):
        return [len(set(_tokens(query)) & set(_tokens(passage))) for query, passage in pairs]


class Reply:
    def __init__(self, content: str):
        self.content = content


class CitingLLM:
    """Answers with the first evidence line and cites its page, or a fixed page if given."""

    def __init__(self, forced_page: int | None = None):
        self.forced_page = forced_page
        self.prompts: list[str] = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        page = self.forced_page or int(re.search(r"\[p\.(\d+) \|", prompt).group(1))
        return Reply(f"Answer from the evidence [p.{page}].")


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page."""
    objects: list[bytes] = []
    page_ids = [3 + 2 * index for index in range(len(pages))]
    font_id = 3 + 2 * len(pages)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    for index, text in enumerate(pages):
        content_id = page_ids[index] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(output)
