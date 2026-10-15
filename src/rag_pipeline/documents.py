"""Documents and text chunking.

Chunking is deliberately boring and deterministic: split into paragraphs,
pack them into windows of ``max_chars`` with ``overlap_chars`` carried over.
Determinism matters — it is what makes the offline evaluation reproducible.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document with a stable id used for citations and eval."""

    id: str
    text: str
    source: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A piece of a document, ready to be embedded."""

    id: str
    doc_id: str
    text: str
    index: int  # position within its document
    source: str = ""


def split_text(text: str, max_chars: int = 800, overlap_chars: int = 100) -> list[str]:
    """Split text into windows of at most ``max_chars`` with overlap.

    Splits prefer paragraph boundaries, then sentence boundaries, then hard
    character cuts. Consecutive windows share ``overlap_chars`` so context is
    not lost at boundaries.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
            continue
        # long paragraph -> sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        buf = ""
        for sent in sentences:
            # a single "sentence" longer than max_chars (no punctuation)
            # needs a hard character cut
            while len(sent) > max_chars:
                if buf:
                    units.append(buf)
                    buf = ""
                units.append(sent[:max_chars])
                sent = sent[max_chars:]
            if buf and len(buf) + 1 + len(sent) > max_chars:
                units.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf)

    chunks: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
        elif len(buf) + 1 + len(unit) <= max_chars:
            buf = f"{buf}\n{unit}"
        else:
            chunks.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail} {unit}".strip()
            if len(buf) > max_chars:
                # keep the new content, drop overlap head until it fits
                buf = buf[-max_chars:]
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(doc: Document, max_chars: int = 800, overlap_chars: int = 100) -> list[Chunk]:
    """Chunk one document; chunk ids are ``<doc_id>#<i>``."""
    texts = split_text(doc.text, max_chars=max_chars, overlap_chars=overlap_chars)
    return [
        Chunk(
            id=f"{doc.id}#{i}",
            doc_id=doc.id,
            text=t,
            index=i,
            source=doc.source,
        )
        for i, t in enumerate(texts)
    ]


def _expand_braces(pattern: str) -> list[str]:
    """Expand ``{a,b}`` groups into separate patterns (pathlib glob does not)."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    head, _, tail = pattern.partition(m.group(0))
    out = []
    for opt in m.group(1).split(","):
        out.extend(_expand_braces(head + opt + tail))
    return out


def load_documents(
    source: str | Path | list[str | Path],
    *,
    pattern: str = "*",
) -> list[Document]:
    """Load text files into Documents.

    Accepts a file, a directory (recursive glob via ``pattern``, with
    ``{a,b}`` brace groups), or a list. Document ids are relative POSIX
    paths (stable and citation-friendly).
    """
    if isinstance(source, (str, Path)):
        source = [source]
    paths: list[Path] = []
    for s in source:
        p = Path(s)
        if p.is_dir():
            for pat in _expand_braces(pattern):
                paths.extend(p.glob(f"**/{pat}"))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rp)
    docs: list[Document] = []
    for p in unique:
        if not p.is_file():
            continue
        rel = p.resolve().as_posix()
        docs.append(Document(id=rel, text=p.read_text(encoding="utf-8"), source=rel))
    return docs


def chunk_documents(
    docs: Iterable[Document], max_chars: int = 800, overlap_chars: int = 100
) -> list[Chunk]:
    """Chunk many documents (fluent helper for the pipeline)."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_text(doc, max_chars=max_chars, overlap_chars=overlap_chars))
    return out
