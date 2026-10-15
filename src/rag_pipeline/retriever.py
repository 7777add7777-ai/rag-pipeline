"""Retriever: query text in, ranked chunks out."""

from __future__ import annotations

from collections.abc import Sequence

from .documents import Chunk
from .embeddings import Embedder
from .vector_store import RankedChunk, VectorStore


class Retriever:
    """Embeds queries and searches the store."""

    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[RankedChunk]:
        """Rank chunks for one query."""
        k = top_k or self.top_k
        if not self.store:
            return []
        vec = self.embedder.embed([query])[0]
        return self.store.search(vec, top_k=k)

    def search_batch(
        self, queries: Sequence[str], top_k: int | None = None
    ) -> list[list[RankedChunk]]:
        """Rank chunks for many queries (embeds in one batch)."""
        if not queries:
            return []
        if not self.store:
            return [[] for _ in queries]
        k = top_k or self.top_k
        vectors = self.embedder.embed(list(queries))
        return [self.store.search(v, top_k=k) for v in vectors]


def build_store(
    chunks: Sequence[Chunk], embedder: Embedder
) -> VectorStore:
    """Embed all chunks and build a store (the pipeline's ingest step)."""
    store = VectorStore()
    if not chunks:
        return store
    vectors = embedder.embed([c.text for c in chunks])
    for chunk, vec in zip(chunks, vectors, strict=False):
        store.add(chunk, vec)
    return store
