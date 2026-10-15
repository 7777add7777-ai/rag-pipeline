"""Answer generators: the pluggable generation layer.

* :class:`TemplateGenerator` — deterministic, offline. Extracts the most
  relevant sentences from the top chunks and cites their sources. Perfect
  for tests, demos, and CI (it makes end-to-end answer quality measurable
  without any API key).
* :class:`OpenAICompatibleGenerator` — any ``/chat/completions`` endpoint.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx

from .vector_store import RankedChunk


class Generator(ABC):
    """Turns a query + retrieved context into an answer."""

    @abstractmethod
    def generate(self, query: str, context: Sequence[RankedChunk]) -> str:
        """Return the answer text for the query given ranked context."""


class TemplateGenerator(Generator):
    """Deterministic extractive generator (offline).

    Picks the up-to-``max_sentences`` sentences across the top chunks that
    share the most content words with the query, and appends a citation line
    with the source ids.
    """

    def __init__(self, max_sentences: int = 2, min_score: float = -1.0):
        self.max_sentences = max_sentences
        self.min_score = min_score

    def _content_words(self, text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}

    def generate(self, query: str, context: Sequence[RankedChunk]) -> str:
        usable = [rc for rc in context if rc.score >= self.min_score]
        if not usable:
            return "No relevant context found."
        qwords = self._content_words(query)

        scored: list[tuple[float, int, str, RankedChunk]] = []
        for rc in usable:
            for i, sentence in enumerate(re.split(r"(?<=[.!?])\s+", rc.chunk.text)):
                if len(sentence.strip()) < 8:
                    continue
                overlap = len(qwords & self._content_words(sentence))
                if overlap:
                    scored.append((overlap * rc.score, i, sentence.strip(), rc))

        if not scored:
            top = usable[0]
            return f"{top.chunk.text[:300]} (source: {top.chunk.source or top.chunk.doc_id})"

        scored.sort(key=lambda t: t[0], reverse=True)
        picked = sorted(scored[: self.max_sentences], key=lambda t: (t[3].chunk.id, t[1]))
        answer = " ".join(t[2] for t in picked)
        sources = sorted({rc.chunk.source or rc.chunk.doc_id for _, _, _, rc in picked})
        return f"{answer} [citations: {', '.join(sources)}]"


class OpenAICompatibleGenerator(Generator):
    """Generates via any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        system_prompt: str = (
            "Answer the question using only the provided context. "
            "Cite the source ids you use."
        ),
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.timeout = timeout
        self._client = http_client or httpx.Client(timeout=timeout)

    def _context_block(self, context: Sequence[RankedChunk]) -> str:
        if not context:
            return "(no context)"
        parts = [
            f"[{rc.chunk.id}] ({rc.chunk.source or rc.chunk.doc_id}) {rc.chunk.text}"
            for rc in context
        ]
        return "\n\n".join(parts)

    def generate(self, query: str, context: Sequence[RankedChunk]) -> str:
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{self._context_block(context)}\n\n"
                            f"Question: {query}\n\nAnswer:"
                        ),
                    },
                ],
                "temperature": 0.0,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"chat API error {resp.status_code}: {resp.text[:200]!r}"
            )
        data = json.loads(resp.text)
        return data["choices"][0]["message"]["content"].strip()
