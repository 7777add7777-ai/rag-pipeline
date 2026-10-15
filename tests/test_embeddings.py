"""Embedder tests: determinism, normalization, ranking, HTTP client."""

import httpx
import pytest

from rag_pipeline.embeddings import (
    DimMismatchError,
    HashEmbedder,
    OpenAICompatibleEmbedder,
    cosine,
)

EMB_OK = {"data": [{"index": 0, "embedding": [0.1, 0.2]}], "model": "m"}


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_hash_embedder_deterministic():
    e = HashEmbedder()
    assert e.embed(["hello world"]) == e.embed(["hello world"])
    assert e.embed(["abc"]) != e.embed(["def"])


def test_hash_embedder_dim_and_normalization():
    e = HashEmbedder(dim=128)
    v = e.embed(["some text about refunds"])[0]
    assert len(v) == 128
    norm = sum(x * x for x in v) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_hash_embedder_empty_text_is_zero_vector():
    e = HashEmbedder()
    assert e.embed([""])[0] == [0.0] * e.dim


def test_hash_embedder_validation():
    with pytest.raises(ValueError, match="dim"):
        HashEmbedder(dim=4)
    with pytest.raises(ValueError, match="ngram"):
        HashEmbedder(ngram=0)


def test_hash_embedder_lexical_ranking():
    e = HashEmbedder()
    query = "refund policy for defective products"
    near = "Customers may request a full refund within 30 days for defective products."
    far = "International shipping is available to 40 countries and takes 10-15 days."
    q = e.embed([query])[0]
    s_near = cosine(q, e.embed([near])[0])
    s_far = cosine(q, e.embed([far])[0])
    assert s_near > s_far
    assert s_near > 0.05


def test_cosine_zero_and_mismatch():
    assert cosine([0, 0], [1, 2]) == 0.0
    with pytest.raises(DimMismatchError):
        cosine([1, 2], [1, 2, 3])


def test_openai_embedder_request_payload():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        import json

        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [0.5, 0.5, 0.5, 0.5]}]}
        )

    p = OpenAICompatibleEmbedder(
        "sk-x", model="emb", base_url="https://api.test/v1", dim=4, http_client=make_client(handler)
    )
    vecs = p.embed(["hello"])
    assert vecs == [[0.5, 0.5, 0.5, 0.5]]
    assert seen["url"] == "https://api.test/v1/embeddings"
    assert seen["auth"] == "Bearer sk-x"
    assert seen["body"]["input"] == ["hello"]


def test_openai_embedder_sorts_by_index():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [9.0, 9.0]},
                    {"index": 0, "embedding": [1.0, 1.0]},
                ]
            },
        )

    p = OpenAICompatibleEmbedder("k", "m", dim=2, http_client=make_client(handler))
    assert p.embed(["a", "b"]) == [[1.0, 1.0], [9.0, 9.0]]


def test_openai_embedder_error_status():
    p = OpenAICompatibleEmbedder(
        "k", "m", http_client=make_client(lambda r: httpx.Response(500, text="boom"))
    )
    with pytest.raises(RuntimeError, match="500"):
        p.embed(["a"])


def test_openai_embedder_dim_mismatch():
    p = OpenAICompatibleEmbedder(
        "k", "m", dim=8, http_client=make_client(lambda r: httpx.Response(200, json=EMB_OK))
    )
    with pytest.raises(DimMismatchError):
        p.embed(["a"])


def test_openai_embedder_empty_input():
    client = make_client(lambda r: httpx.Response(200))
    p = OpenAICompatibleEmbedder("k", "m", dim=2, http_client=client)
    assert p.embed([]) == []
