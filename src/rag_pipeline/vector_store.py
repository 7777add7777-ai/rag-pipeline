"""In-memory vector store with cosine similarity.

Pure Python on purpose: the retrieval math is O(N*D) which is fine for
thousands of chunks and keeps the package dependency-light. Swap this class
for FAISS/pgvector when you outgrow it — the :class:`Retriever` interface
stays the same.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .documents import Chunk
from .embeddings import cosine


class RankedChunk(BaseModel):
    """A chunk with its retrieval score."""

    chunk: Chunk
    score: float


class VectorStore:
    """Flat list of (id, vector, chunk) with linear cosine scan."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._chunks: dict[str, Chunk] = {}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int | None:
        return len(self._vectors[0]) if self._vectors else None

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        if chunk.id in self._chunks:
            raise ValueError(f"chunk id already present: {chunk.id}")
        if self._vectors and len(vector) != len(self._vectors[0]):
            raise ValueError(
                f"dimension mismatch: store has {len(self._vectors[0])}, "
                f"got {len(vector)}"
            )
        self._ids.append(chunk.id)
        self._vectors.append(vector)
        self._chunks[chunk.id] = chunk

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RankedChunk]:
        """Return the top-k most similar chunks, best first."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        scored = [
            RankedChunk(chunk=self._chunks[cid], score=cosine(query_vector, vec))
            for cid, vec in zip(self._ids, self._vectors, strict=False)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        """Persist the store as JSON (vectors rounded for readability)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": {cid: c.model_dump() for cid, c in self._chunks.items()},
            "order": self._ids,
            "vectors": self._vectors,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        chunks = {cid: Chunk.model_validate(c) for cid, c in payload["chunks"].items()}
        for cid in payload["order"]:
            store.add(chunks[cid], payload["vectors"][payload["order"].index(cid)])
        return store
