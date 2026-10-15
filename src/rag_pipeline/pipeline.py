"""RAGPipeline: ingest -> retrieve -> answer, as one object."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from .documents import Chunk, Document, chunk_documents
from .embeddings import Embedder
from .generator import Generator
from .retriever import Retriever, build_store
from .vector_store import RankedChunk, VectorStore


class Answer(BaseModel):
    """Result of :meth:`RAGPipeline.answer`."""

    query: str
    answer: str
    context: list[RankedChunk]


class RAGPipeline:
    """Ties chunking, embedding, storage, retrieval, and generation together.

    Typical use::

        pipeline = RAGPipeline(HashEmbedder())
        pipeline.ingest(load_documents("kb/"))
        answer = pipeline.answer("what is the refund window?")
    """

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator | None = None,
        *,
        top_k: int = 5,
        max_chunk_chars: int = 800,
        overlap_chars: int = 100,
    ):
        from .generator import TemplateGenerator

        self.embedder = embedder
        self.generator = generator or TemplateGenerator()
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.store = VectorStore()
        self.retriever = Retriever(self.store, embedder, top_k=top_k)
        self.n_documents = 0

    def ingest(
        self,
        documents: Iterable[Document],
        *,
        max_chunk_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> int:
        """Embed and store all chunks of the given documents; returns chunk count."""
        max_chars = max_chunk_chars or self.max_chunk_chars
        overlap = overlap_chars or self.overlap_chars
        chunks: list[Chunk] = chunk_documents(
            documents, max_chars=max_chars, overlap_chars=overlap
        )
        if not chunks:
            return 0
        new_store = build_store(chunks, self.embedder)
        # rebuild: keeps the store consistent even if ids were re-ingested
        self.store = new_store
        self.retriever.store = new_store
        self.n_documents += len({c.doc_id for c in chunks})
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RankedChunk]:
        return self.retriever.search(query, top_k=top_k)

    def answer(self, query: str, top_k: int | None = None) -> Answer:
        """Retrieve context and generate an answer with citations."""
        context = self.retrieve(query, top_k=top_k)
        text = self.generator.generate(query, context)
        return Answer(query=query, answer=text, context=context)

    def ask_batch(
        self, queries: Sequence[str], top_k: int | None = None
    ) -> list[Answer]:
        """Answer many queries (batched embedding for the retrieval part)."""
        if not queries:
            return []
        k = top_k or self.top_k
        if not self.store:
            return [Answer(query=q, answer="No relevant context found.", context=[])
                    for q in queries]
        vectors = self.embedder.embed(list(queries))
        out: list[Answer] = []
        for q, vec in zip(queries, vectors, strict=False):
            ranked = self.store.search(vec, top_k=k)
            out.append(
                Answer(query=q, answer=self.generator.generate(q, ranked), context=ranked)
            )
        return out
