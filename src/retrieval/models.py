"""Pydantic v2 data models for retrieval layer."""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    HIERARCHICAL = "hierarchical"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC = "semantic"


class Document(BaseModel):
    """A document with metadata and embedding."""

    document_id: str = Field(
        ..., description="Unique identifier for the document"
    )
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata (source, date, etc)"
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="768-dim embedding vector"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    source: str = Field(default="", description="Source URL or path")
    chunk_count: int = Field(default=0, description="Number of chunks in document")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return f"Document(id={self.document_id}, title={self.title[:50]}, chunks={self.chunk_count})"


class Chunk(BaseModel):
    """A chunk of text with metadata."""

    chunk_id: str = Field(
        ..., description="Unique chunk identifier (doc_id::chunk_num)"
    )
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Number of tokens in chunk")
    embedding: Optional[list[float]] = Field(
        default=None, description="768-dim embedding vector"
    )
    position: int = Field(default=0, description="Position in document (0-indexed)")
    is_parent: bool = Field(
        default=False, description="True if this is a parent (512-token) chunk"
    )
    parent_id: Optional[str] = Field(
        default=None, description="Parent chunk ID if this is a child"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Chunk-level metadata"
    )

    @validator("chunk_id")
    def validate_chunk_id(cls, v: str) -> str:
        """Validate chunk ID format."""
        if "::" not in v:
            raise ValueError("chunk_id must be in format 'document_id::chunk_num'")
        return v

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return f"Chunk(id={self.chunk_id}, tokens={self.token_count}, parent={self.is_parent})"


class HyDeResult(BaseModel):
    """Result from HyDE expansion."""

    query: str = Field(..., description="Original query")
    hypotheticals: list[str] = Field(
        ..., description="3 hypothetical documents answering query"
    )
    query_embedding: list[float] = Field(
        ..., description="Embedding of original query"
    )
    hypothetical_embeddings: list[list[float]] = Field(
        ..., description="Embeddings of 3 hypothetical documents"
    )
    averaged_embedding: list[float] = Field(
        ..., description="Average of query + hypothetical embeddings"
    )
    generation_latency_ms: float = Field(
        default=0.0, description="Latency for generating hypotheticals"
    )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return f"HyDeResult(query={self.query[:50]}, hypotheticals={len(self.hypotheticals)})"


class SearchResult(BaseModel):
    """A single search result."""

    chunk: Chunk = Field(..., description="Retrieved chunk")
    score: float = Field(..., description="Relevance score (0.0-1.0)")
    rank: int = Field(default=0, description="Rank in result set")
    bm25_rank: Optional[int] = Field(default=None, description="BM25 ranking")
    vector_rank: Optional[int] = Field(default=None, description="Vector ranking")
    reranker_score: Optional[float] = Field(
        default=None, description="Cross-encoder reranking score"
    )
    retrieval_method: str = Field(
        default="hybrid", description="How this was retrieved (hybrid, bm25, vector)"
    )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return f"SearchResult(chunk={self.chunk.chunk_id}, score={self.score:.3f}, rank={self.rank})"


class ChunkerConfig(BaseModel):
    """Configuration for hierarchical chunking."""

    parent_chunk_size: int = Field(
        default=512, description="Tokens per parent chunk"
    )
    child_chunk_size: int = Field(
        default=128, description="Tokens per child chunk"
    )
    overlap_ratio: float = Field(
        default=0.2, description="Overlap between chunks (0.0-1.0)"
    )
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.HIERARCHICAL, description="Chunking strategy"
    )
    min_chunk_size: int = Field(
        default=50, description="Minimum tokens in a chunk"
    )
    semantic_similarity_threshold: float = Field(
        default=0.8, description="Threshold for semantic boundaries (0.0-1.0)"
    )

    @validator("overlap_ratio")
    def validate_overlap(cls, v: float) -> float:
        """Validate overlap ratio."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("overlap_ratio must be between 0.0 and 1.0")
        return v

    @validator("semantic_similarity_threshold")
    def validate_threshold(cls, v: float) -> float:
        """Validate similarity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("semantic_similarity_threshold must be between 0.0 and 1.0")
        return v

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return f"ChunkerConfig(parent={self.parent_chunk_size}, child={self.child_chunk_size}, strategy={self.strategy})"


class RerankerConfig(BaseModel):
    """Configuration for cross-encoder reranking."""

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model name from HuggingFace",
    )
    batch_size: int = Field(default=32, description="Batch size for inference")
    top_k: int = Field(default=5, description="Return top-k results after reranking")
    input_max_length: int = Field(
        default=512, description="Maximum input length for model"
    )
    threshold: Optional[float] = Field(
        default=None, description="Minimum score threshold (optional)"
    )
    use_gpu: bool = Field(default=False, description="Use GPU if available")

    @validator("top_k")
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k."""
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return f"RerankerConfig(model={self.model_name.split('/')[-1]}, top_k={self.top_k})"


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search (BM25 + Vector)."""

    alpha: float = Field(
        default=0.5,
        description="Weight for BM25 vs vector (0=pure vector, 1=pure BM25)"
    )
    top_k: int = Field(default=20, description="Number of results to return")
    bm25_k1: float = Field(
        default=1.5, description="BM25 k1 parameter (term frequency saturation)"
    )
    bm25_b: float = Field(
        default=0.75, description="BM25 b parameter (length normalization)"
    )
    rrf_constant: int = Field(
        default=60, description="RRF constant for combining rankings"
    )
    deduplicate: bool = Field(
        default=True, description="Remove duplicate documents"
    )
    normalize_scores: bool = Field(
        default=True, description="Normalize scores to 0-1 range"
    )

    @validator("alpha")
    def validate_alpha(cls, v: float) -> float:
        """Validate alpha."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return f"HybridSearchConfig(alpha={self.alpha}, top_k={self.top_k})"


@dataclass
class EmbeddingRequest:
    """Request for embeddings."""
    text: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    task_prefix: str = "search_document:"


@dataclass
class TokenStats:
    """Token count statistics."""
    query_tokens: int
    chunk_tokens: int
    total_tokens: int
    embedding_cost_tokens: int
