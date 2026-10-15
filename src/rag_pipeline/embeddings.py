"""Embedders: the pluggable vectorization layer.

Two implementations ship in the box:

* :class:`HashEmbedder` — deterministic, dependency-free, fully offline.
  Character n-gram hashing into a fixed-size vector with L2 normalization.
  It captures lexical overlap (a good-enough stand-in for testing, demos,
  and CI) and makes evaluation fully reproducible.
* :class:`OpenAICompatibleEmbedder` — talks to any
  ``/embeddings``-compatible HTTP API (OpenAI, Azure, vLLM, Ollama, ...).

Real semantic quality comes from a real model; this package's job is to make
swapping them a one-line change.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx


class DimMismatchError(ValueError):
    """Raised when vectors of different dimensions are compared."""


class Embedder(ABC):
    """Vectorizes text. Subclasses must be deterministic per input."""

    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""


class HashEmbedder(Embedder):
    """Deterministic n-gram hash embedder (offline, no dependencies).

    Each character trigram (plus unigrams, weighted) is hashed to a bucket
    and a sign, accumulating into a ``dim``-sized vector that is then L2
    normalized. Texts sharing tokens share vector mass -> cosine similarity
    tracks lexical overlap, which is enough to test retrieval end to end.
    """

    def __init__(self, dim: int = 384, ngram: int = 3):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        if ngram < 1:
            raise ValueError("ngram must be >= 1")
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> list[tuple[str, float]]:
        t = text.lower().strip()
        feats: dict[str, float] = {}
        for token in re.findall(r"[a-z0-9]+", t):
            feats[token] = feats.get(token, 0.0) + 1.0
        if len(t) >= self.ngram:
            for i in range(len(t) - self.ngram + 1):
                gram = t[i : i + self.ngram]
                feats[gram] = feats.get(gram, 0.0) + 0.5
        return list(feats.items())

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for feat, weight in self._features(text):
            digest = hashlib.md5(feat.encode("utf-8")).digest()  # noqa: S324
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            v[idx] += sign * weight
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            v = [x / norm for x in v]
        return v

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAICompatibleEmbedder(Embedder):
    """Embeds via any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.timeout = timeout
        self._client = http_client or httpx.Client(timeout=timeout)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"embeddings API error {resp.status_code}: {resp.text[:200]!r}")
        data = resp.json()
        rows = sorted(data["data"], key=lambda r: r["index"])
        vectors = [list(r["embedding"]) for r in rows]
        if vectors and len(vectors[0]) != self.dim:
            raise DimMismatchError(
                f"model returned {len(vectors[0])} dims, expected {self.dim}"
            )
        return vectors


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; raises :class:`DimMismatchError` on shape mismatch."""
    if len(a) != len(b):
        raise DimMismatchError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
