"""Retriever tests."""

from rag_pipeline.documents import Document, chunk_text
from rag_pipeline.embeddings import HashEmbedder
from rag_pipeline.retriever import Retriever, build_store


def make_kb():
    docs = [
        Document(id="refund", text="Full refund within 30 days for defective products."),
        Document(id="shipping", text="Express shipping costs 9.99 and takes 1-2 days."),
    ]
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d))
    return build_store(chunks, HashEmbedder())


def test_retriever_search_ranks():
    store = make_kb()
    r = Retriever(store, HashEmbedder(), top_k=2)
    out = r.search("refund for defective products")
    assert [c.chunk.doc_id for c in out] == ["refund", "shipping"]
    assert out[0].score > out[1].score


def test_retriever_empty_store():
    r = Retriever(build_store([], HashEmbedder()), HashEmbedder())
    assert r.search("anything") == []
    assert r.search_batch(["a", "b"]) == [[], []]


def test_retriever_batch_matches_single():
    store = make_kb()
    r = Retriever(store, HashEmbedder(), top_k=2)
    batch = r.search_batch(["refund window", "shipping cost"])
    assert len(batch) == 2
    single = r.search("shipping cost")
    assert [c.chunk.id for c in batch[1]] == [c.chunk.id for c in single]


def test_retriever_batch_empty_queries():
    r = Retriever(make_kb(), HashEmbedder())
    assert r.search_batch([]) == []


def test_build_store_empty():
    store = build_store([], HashEmbedder())
    assert len(store) == 0


def test_build_store_counts():
    store = make_kb()
    assert len(store) == 2
    assert store.dim == HashEmbedder().dim
