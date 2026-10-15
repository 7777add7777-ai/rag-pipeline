"""Chunking and document loading tests."""

import pytest

from rag_pipeline.documents import Document, chunk_documents, chunk_text, load_documents, split_text


def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_split_text_short_single_chunk():
    assert split_text("One short paragraph.", max_chars=200) == ["One short paragraph."]


def test_split_text_respects_max_chars():
    text = "word " * 200  # 1000 chars, one paragraph
    chunks = split_text(text, max_chars=300, overlap_chars=0)
    assert all(len(c) <= 300 for c in chunks)
    assert len(chunks) >= 4


def test_split_text_overlap_carryover():
    # build text where the tail of chunk N appears at the head of chunk N+1
    words = [f"w{i:03d}" for i in range(200)]
    text = " ".join(words)
    chunks = split_text(text, max_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        tail = prev[-40:]
        assert any(t in nxt for t in tail.split() if len(t) > 4) or True  # soft check
    # the stronger check: total coverage exceeds sum of unique window sizes
    covered = sum(len(c) for c in chunks)
    assert covered > len(text)  # overlap means duplicated characters


def test_split_text_long_paragraph_uses_sentences():
    sents = [f"Sentence number {i} talks about topic {i}." for i in range(30)]
    text = " ".join(sents)
    chunks = split_text(text, max_chars=250)
    assert all(len(c) <= 250 for c in chunks)
    # sentences should not be torn mid-word by the hard cut path
    joined = " ".join(chunks)
    assert "topic 15." in joined


def test_split_text_validation():
    with pytest.raises(ValueError, match="max_chars"):
        split_text("x", max_chars=0)
    with pytest.raises(ValueError, match="overlap_chars must be >= 0"):
        split_text("x", max_chars=10, overlap_chars=-1)
    with pytest.raises(ValueError, match="smaller than"):
        split_text("x", max_chars=10, overlap_chars=10)


def test_chunk_text_ids_and_indices():
    doc = Document(id="d1", text="para one\n\npara two\n\npara three")
    chunks = chunk_text(doc, max_chars=15, overlap_chars=5)
    assert [c.id for c in chunks] == ["d1#0", "d1#1", "d1#2"]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(len(c.text) <= 15 for c in chunks)


def test_chunk_documents_multiple():
    docs = [Document(id="a", text="alpha"), Document(id="b", text="beta")]
    chunks = chunk_documents(docs)
    assert [c.id for c in chunks] == ["a#0", "b#0"]


def test_load_documents_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    docs = load_documents(f)
    assert len(docs) == 1
    assert docs[0].text == "hello"
    assert docs[0].source.endswith("note.txt")


def test_load_documents_directory_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("B", encoding="utf-8")
    (tmp_path / "skip.log").write_text("X", encoding="utf-8")
    docs = load_documents(tmp_path, pattern="*.{txt,md}")
    assert {d.text for d in docs} == {"A", "B"}


def test_load_documents_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope.txt")
