# rag-pipeline

A dependency-light RAG pipeline in Python: **chunking → embeddings → vector
search → answer generation**, with a **fully offline evaluation harness**.

```
query ──> embed ──> cosine search ──> top-k chunks ──> generator ──> answer
```

## Why

Most RAG tutorials hide the unglamorous 90%: chunking strategy, retrieval
quality, and evaluation. This package makes that 90% explicit and testable:

- **Pluggable everything.** `Embedder`, `Generator`, `VectorStore` are small
  interfaces. Ship the deterministic offline implementations; swap in real
  models when you have them — the rest of the code does not change.
- **Deterministic offline core.** `HashEmbedder` + `TemplateGenerator` need
  no API key and no network, so the whole pipeline (including evaluation)
  runs in CI in under a second.
- **Evaluation first.** `GoldenCase` → `eval_retrieval` →
  recall@k / MRR / hit rate, plus answer substring checks. You measure
  retrieval quality before you ever spend a token on generation.

## Quickstart

```bash
pip install "rag-pipeline[dev]"
ragpy demo          # offline demo: KB ingestion + retrieval & answer eval
```

```python
from rag_pipeline import (
    RAGPipeline, HashEmbedder, TemplateGenerator, load_documents,
)

pipeline = RAGPipeline(HashEmbedder(), TemplateGenerator(), top_k=3)
pipeline.ingest(load_documents("kb/"))

answer = pipeline.answer("What is the refund window?")
print(answer.answer)        # extractive answer with citations
print([c.chunk.id for c in answer.context])
```

Real models are a one-line swap:

```python
from rag_pipeline import OpenAICompatibleEmbedder, OpenAICompatibleGenerator

pipeline = RAGPipeline(
    OpenAICompatibleEmbedder(api_key=..., model="text-embedding-3-small", dim=1536),
    OpenAICompatibleGenerator(api_key=..., model="gpt-4o-mini"),
    top_k=5,
)
```

## Pipeline pieces

| Piece | Class | Notes |
|---|---|---|
| Documents & chunking | `Document`, `chunk_text` | paragraph-aware windows with overlap |
| Embeddings | `HashEmbedder`, `OpenAICompatibleEmbedder` | deterministic offline / any `/embeddings` API |
| Vector store | `VectorStore` | flat cosine search, JSON persistence |
| Retrieval | `Retriever` | single + batched query search |
| Generation | `TemplateGenerator`, `OpenAICompatibleGenerator` | extractive offline / any `/chat/completions` API |
| Pipeline | `RAGPipeline` | ingest → retrieve → answer, citations |
| Evaluation | `GoldenCase`, `eval_retrieval`, `eval_answers` | recall@k, MRR, hit rate, answer checks |

## Testing

```bash
python -m pytest   # 50+ offline tests, no network
ruff check .
```

## Architecture notes

- The vector store is deliberately a flat pure-Python cosine scan: O(N·D)
  is fine for thousands of chunks, and the `Retriever` interface is what you
  keep when you move to FAISS or pgvector.
- Chunk ids (`<doc_id>#<i>`) are stable strings used for citations and as
  the ground-truth keys in eval.
- `HashEmbedder` measures lexical overlap, not semantics — use it to test
  the plumbing; use a real model to measure real quality.

## License

MIT — see [LICENSE](LICENSE).
