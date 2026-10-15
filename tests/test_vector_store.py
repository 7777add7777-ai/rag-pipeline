"""Vector store tests."""

import pytest

from rag_pipeline.documents import Chunk
from rag_pipeline.vector_store import VectorStore


def make_chunk(cid: str, text: str = "t") -> Chunk:
    return Chunk(id=cid, doc_id="d", text=text, index=0)


def test_add_and_get():
    s = VectorStore()
    s.add(make_chunk("a"), [1.0, 0.0])
    assert len(s) == 1
    assert s.get("a").id == "a"
    assert s.dim == 2


def test_add_duplicate_id_raises():
    s = VectorStore()
    s.add(make_chunk("a"), [1.0, 0.0])
    with pytest.raises(ValueError, match="already present"):
        s.add(make_chunk("a"), [0.0, 1.0])


def test_add_dim_mismatch_raises():
    s = VectorStore()
    s.add(make_chunk("a"), [1.0, 0.0])
    with pytest.raises(ValueError, match="dimension mismatch"):
        s.add(make_chunk("b"), [1.0, 0.0, 0.0])


def test_empty_store():
    s = VectorStore()
    assert len(s) == 0
    assert s.dim is None
    assert s.search([1.0], top_k=1) == []


def test_search_ranking():
    s = VectorStore()
    s.add(make_chunk("x"), [1.0, 0.0])
    s.add(make_chunk("y"), [0.0, 1.0])
    s.add(make_chunk("z"), [0.7, 0.7])
    results = s.search([1.0, 0.1], top_k=2)
    assert [r.chunk.id for r in results] == ["x", "z"]
    assert results[0].score > results[1].score


def test_search_top_k_validation():
    s = VectorStore()
    s.add(make_chunk("a"), [1.0])
    with pytest.raises(ValueError, match="top_k"):
        s.search([1.0], top_k=0)


def test_search_returns_all_when_top_k_exceeds():
    s = VectorStore()
    s.add(make_chunk("a"), [1.0, 0.0])
    s.add(make_chunk("b"), [0.0, 1.0])
    assert len(s.search([1.0, 0.0], top_k=10)) == 2


def test_save_load_roundtrip(tmp_path):
    s = VectorStore()
    s.add(make_chunk("a", "alpha text"), [0.6, 0.8])
    s.add(make_chunk("b", "beta text"), [0.8, 0.6])
    path = tmp_path / "store.json"
    s.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    assert loaded.get("a").text == "alpha text"
    assert [r.chunk.id for r in s.search([1.0, 0.0], top_k=2)] == [
        r.chunk.id for r in loaded.search([1.0, 0.0], top_k=2)
    ]


def test_save_creates_parent_dirs(tmp_path):
    s = VectorStore()
    s.add(make_chunk("a"), [1.0])
    s.save(tmp_path / "nested" / "deep" / "store.json")
    assert (tmp_path / "nested" / "deep" / "store.json").exists()


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "missing.json")
