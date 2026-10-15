"""Evaluation metrics tests."""

import pytest

from rag_pipeline.documents import Document, chunk_text
from rag_pipeline.embeddings import HashEmbedder
from rag_pipeline.evaluation import (
    GoldenCase,
    eval_answers,
    eval_retrieval,
    mrr,
    recall_at_k,
)
from rag_pipeline.retriever import Retriever, build_store


def make_retriever():
    docs = [
        Document(id="a", text="Refunds are available within 30 days for defective products."),
        Document(id="b", text="Express shipping costs 9.99 dollars and arrives in two days."),
        Document(id="c", text="Warranty excludes batteries filters and consumables from coverage."),
    ]
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d))
    store = build_store(chunks, HashEmbedder())
    return Retriever(store, HashEmbedder(), top_k=3)


def test_recall_at_k_basic():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0
    assert recall_at_k(["x", "y"], {"a"}, k=5) == 0.0
    assert recall_at_k([], set(), k=1) == 0.0


def test_mrr_basic():
    assert mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)
    assert mrr(["a"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_eval_retrieval_perfect():
    r = make_retriever()
    cases = [
        GoldenCase(query="refund defective days", expected_chunk_ids=("a#0",)),
        GoldenCase(query="express shipping cost", expected_chunk_ids=("b#0",)),
    ]
    rep = eval_retrieval(r, cases, ks=(1, 3))
    assert rep.n_cases == 2
    assert rep.recall_at_k[1] == pytest.approx(1.0)
    assert rep.mrr == pytest.approx(1.0)
    assert rep.hit_rate == pytest.approx(1.0)
    assert "R@1" in rep.summary()


def test_eval_retrieval_partial_hit():
    r = make_retriever()
    # query matching 'a' but expecting 'c' -> c lands at rank 3
    rep = eval_retrieval(
        r,
        [GoldenCase(query="refund defective", expected_chunk_ids=("c#0",))],
        ks=(1,),
    )
    assert rep.recall_at_k[1] == 0.0
    assert rep.mrr == pytest.approx(1 / 3)
    assert rep.hit_rate == pytest.approx(1.0)


def test_eval_retrieval_empty_cases_raises():
    r = make_retriever()
    with pytest.raises(ValueError, match="no golden cases"):
        eval_retrieval(r, [])


def test_eval_answers_hits_and_misses():
    cases = [
        GoldenCase(query="q1", expected_chunk_ids=("a",), expected_answer_contains=("30 days",)),
        GoldenCase(
            query="q2", expected_chunk_ids=("b",), expected_answer_contains=("9.99", "NOPE")
        ),
    ]
    rep = eval_answers([("Refunds within 30 days.", cases[0]), ("Cost is 9.99.", cases[1])])
    assert rep.n_cases == 2
    assert rep.exact_hits == 1
    assert rep.hit_rate == pytest.approx(0.5)
    miss = [r for r in rep.per_case if not r["ok"]][0]
    assert miss["missing"] == ["NOPE"]


def test_eval_answers_no_expectations():
    cases = [GoldenCase(query="q", expected_chunk_ids=("a",))]
    rep = eval_answers([("anything", cases[0])])
    assert rep.n_cases == 1
    assert rep.hit_rate == 0.0
    assert rep.per_case == []


def test_eval_answers_empty_raises():
    with pytest.raises(ValueError, match="no answers"):
        eval_answers([])


def test_eval_retrieval_case_insensitive_answer_check():
    cases = [
        GoldenCase(
            query="q", expected_chunk_ids=("a",), expected_answer_contains=("REFUNDS",)
        )
    ]
    rep = eval_answers([("refunds are fast.", cases[0])])
    assert rep.hit_rate == pytest.approx(1.0)
