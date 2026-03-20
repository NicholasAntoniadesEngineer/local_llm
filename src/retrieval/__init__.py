"""Retrieval layer: HyDE, hybrid search, reranking, and hierarchical chunking."""

from .models import (
    Document,
    Chunk,
    HyDeResult,
    SearchResult,
    ChunkerConfig,
    RerankerConfig,
    HybridSearchConfig,
    ChunkingStrategy,
    EmbeddingRequest,
    TokenStats,
)

from .hyde import HyDEExpander

from .hybrid import HybridSearcher

from .reranker import CrossEncoderReranker

from .chunker import HierarchicalChunker


__all__ = [
    # Models
    "Document",
    "Chunk",
    "HyDeResult",
    "SearchResult",
    "ChunkerConfig",
    "RerankerConfig",
    "HybridSearchConfig",
    "ChunkingStrategy",
    "EmbeddingRequest",
    "TokenStats",
    # Components
    "HyDEExpander",
    "HybridSearcher",
    "CrossEncoderReranker",
    "HierarchicalChunker",
]
