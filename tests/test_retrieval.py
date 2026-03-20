"""Tests for retrieval layer (HyDE, hybrid search, reranking, chunking)."""

import pytest
import math
from typing import List

from src.retrieval.models import (
    Document,
    Chunk,
    SearchResult,
    HyDeResult,
    ChunkerConfig,
    HybridSearchConfig,
    RerankerConfig,
    ChunkingStrategy,
)


@pytest.mark.unit
class TestChunkingStrategy:
    """Test document chunking strategies."""

    def test_hierarchical_chunking_config(self):
        """Test hierarchical chunking configuration."""
        config = ChunkerConfig(
            parent_chunk_size=512,
            child_chunk_size=128,
            overlap_ratio=0.2,
            strategy=ChunkingStrategy.HIERARCHICAL,
        )
        assert config.parent_chunk_size == 512
        assert config.child_chunk_size == 128
        assert config.overlap_ratio == 0.2
        assert config.strategy == ChunkingStrategy.HIERARCHICAL

    def test_chunking_overlap_ratio_validation_too_low(self):
        """Test overlap ratio validation rejects negative values."""
        with pytest.raises(ValueError):
            ChunkerConfig(overlap_ratio=-0.1)

    def test_chunking_overlap_ratio_validation_too_high(self):
        """Test overlap ratio validation rejects values > 1."""
        with pytest.raises(ValueError):
            ChunkerConfig(overlap_ratio=1.5)

    def test_reranker_top_k_validation(self):
        """Test top_k validation rejects invalid values."""
        # Valid
        config = RerankerConfig(top_k=10)
        assert config.top_k == 10

        # Invalid - zero
        with pytest.raises(ValueError):
            RerankerConfig(top_k=0)

        # Invalid - negative
        with pytest.raises(ValueError):
            RerankerConfig(top_k=-1)

    def test_hybrid_search_config(self):
        """Test hybrid search configuration."""
        config = HybridSearchConfig(
            alpha=0.5,
            top_k=20,
            bm25_k1=1.5,
            bm25_b=0.75,
            rrf_constant=60,
            deduplicate=True,
            normalize_scores=True,
        )
        assert config.alpha == 0.5
        assert config.top_k == 20
        assert config.deduplicate is True
