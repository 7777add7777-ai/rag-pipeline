"""rag-pipeline: chunking, embeddings, vector search, and offline evaluation."""

from .documents import Chunk, Document, chunk_text, load_documents, split_text
from .embeddings import DimMismatchError, Embedder, HashEmbedder, OpenAICompatibleEmbedder
from .evaluation import (
    AnswerReport,
    GoldenCase,
    RetrievalReport,
    eval_retrieval,
    mrr,
    recall_at_k,
)
from .generator import Generator, OpenAICompatibleGenerator, TemplateGenerator
from .pipeline import RAGPipeline
from .retriever import RankedChunk, Retriever
from .vector_store import VectorStore

__version__ = "0.1.0"

__all__ = [
    "AnswerReport",
    "Chunk",
    "DimMismatchError",
    "Document",
    "Embedder",
    "Generator",
    "GoldenCase",
    "HashEmbedder",
    "OpenAICompatibleEmbedder",
    "OpenAICompatibleGenerator",
    "RAGPipeline",
    "RankedChunk",
    "RetrievalReport",
    "Retriever",
    "TemplateGenerator",
    "VectorStore",
    "chunk_text",
    "eval_retrieval",
    "load_documents",
    "mrr",
    "recall_at_k",
    "split_text",
    "__version__",
]
