"""Tests for the retrieval layer."""

import pytest
from src.retrieval import (
    Document,
    Chunk,
    SearchResult,
    ChunkerConfig,
    RerankerConfig,
    HybridSearchConfig,
    HyDeResult,
)


class TestModels:
    """Test Pydantic models."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            chunk_id="doc1::0",
            document_id="doc1",
            text="This is a test chunk.",
            token_count=5,
            position=0,
            is_parent=True,
        )

        assert chunk.chunk_id == "doc1::0"
        assert chunk.document_id == "doc1"
        assert chunk.token_count == 5
        assert chunk.is_parent is True

    def test_chunk_invalid_id(self):
        """Test chunk with invalid ID format."""
        with pytest.raises(ValueError):
            Chunk(
                chunk_id="invalid",  # Missing ::
                document_id="doc1",
                text="Test",
                token_count=1,
            )

    def test_search_result_creation(self):
        """Test creating a search result."""
        chunk = Chunk(
            chunk_id="doc1::0",
            document_id="doc1",
            text="Test content",
            token_count=2,
        )

        result = SearchResult(
            chunk=chunk,
            score=0.95,
            rank=1,
            retrieval_method="hybrid",
        )

        assert result.score == 0.95
        assert result.rank == 1
        assert result.chunk.chunk_id == "doc1::0"

    def test_chunker_config_validation(self):
        """Test ChunkerConfig validation."""
        # Valid config
        config = ChunkerConfig(
            parent_chunk_size=512,
            child_chunk_size=128,
            overlap_ratio=0.2,
        )
        assert config.overlap_ratio == 0.2

        # Invalid overlap
        with pytest.raises(ValueError):
            ChunkerConfig(overlap_ratio=1.5)

    def test_reranker_config_creation(self):
        """Test RerankerConfig."""
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=32,
            top_k=5,
        )

        assert config.top_k == 5
        assert config.batch_size == 32

    def test_reranker_config_invalid_top_k(self):
        """Test RerankerConfig with invalid top_k."""
        with pytest.raises(ValueError):
            RerankerConfig(top_k=0)

    def test_hybrid_search_config(self):
        """Test HybridSearchConfig."""
        config = HybridSearchConfig(
            alpha=0.5,
            top_k=20,
            bm25_k1=1.5,
            bm25_b=0.75,
        )

        assert config.alpha == 0.5
        assert config.top_k == 20

    def test_hybrid_search_config_invalid_alpha(self):
        """Test HybridSearchConfig with invalid alpha."""
        with pytest.raises(ValueError):
            HybridSearchConfig(alpha=1.5)

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            document_id="doc1",
            title="Test Document",
            content="This is test content.",
            source="http://example.com",
        )

        assert doc.document_id == "doc1"
        assert doc.title == "Test Document"
        assert doc.source == "http://example.com"

    def test_hyde_result_creation(self):
        """Test HyDeResult creation."""
        result = HyDeResult(
            query="What is machine learning?",
            hypotheticals=[
                "ML is a subset of AI",
                "ML enables pattern recognition",
                "ML powers recommendation systems",
            ],
            query_embedding=[0.1] * 768,
            hypothetical_embeddings=[[0.2] * 768, [0.3] * 768, [0.4] * 768],
            averaged_embedding=[0.25] * 768,
        )

        assert len(result.hypotheticals) == 3
        assert len(result.averaged_embedding) == 768


class TestHybridSearcher:
    """Test hybrid searcher."""

    def test_hybrid_searcher_creation(self):
        """Test creating a hybrid searcher."""
        from src.retrieval import HybridSearcher

        searcher = HybridSearcher(chunks=[])
        assert len(searcher.chunks) == 0

    def test_add_chunks(self):
        """Test adding chunks to searcher."""
        from src.retrieval import HybridSearcher

        searcher = HybridSearcher()

        chunks = [
            Chunk(
                chunk_id="doc1::0",
                document_id="doc1",
                text="First chunk about machine learning",
                token_count=5,
                embedding=[0.1] * 768,
            ),
            Chunk(
                chunk_id="doc1::1",
                document_id="doc1",
                text="Second chunk about neural networks",
                token_count=5,
                embedding=[0.2] * 768,
            ),
        ]

        searcher.add_chunks(chunks)
        assert len(searcher.chunks) == 2

    def test_duplicate_chunk_ids(self):
        """Test that duplicate chunk IDs are rejected."""
        from src.retrieval import HybridSearcher

        searcher = HybridSearcher()

        chunk1 = Chunk(
            chunk_id="doc1::0",
            document_id="doc1",
            text="First chunk",
            token_count=2,
        )

        searcher.add_chunks([chunk1])

        with pytest.raises(ValueError, match="Duplicate"):
            searcher.add_chunks([chunk1])

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from src.retrieval import HybridSearcher

        searcher = HybridSearcher()

        # Identical vectors should have similarity 1.0 (for normalized)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = searcher._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

    def test_normalize_scores(self):
        """Test score normalization."""
        from src.retrieval import HybridSearcher

        searcher = HybridSearcher()

        chunk1 = Chunk(
            chunk_id="doc1::0",
            document_id="doc1",
            text="Test",
            token_count=1,
        )

        results = [
            SearchResult(chunk=chunk1, score=0.2),
            SearchResult(chunk=chunk1, score=0.5),
            SearchResult(chunk=chunk1, score=0.8),
        ]

        normalized = searcher._normalize_scores(results)

        # After normalization, scores should be 0.0, 0.5, 1.0
        assert normalized[0].score == 0.0
        assert abs(normalized[1].score - 0.5) < 0.001
        assert normalized[2].score == 1.0


class TestChunker:
    """Test hierarchical chunker."""

    def test_chunker_creation(self):
        """Test creating a chunker."""
        from src.retrieval import HierarchicalChunker

        chunker = HierarchicalChunker()
        assert chunker.config.parent_chunk_size == 512
        assert chunker.config.child_chunk_size == 128

    def test_chunker_config(self):
        """Test chunker with custom config."""
        from src.retrieval import HierarchicalChunker, ChunkerConfig

        config = ChunkerConfig(
            parent_chunk_size=256,
            child_chunk_size=64,
        )

        chunker = HierarchicalChunker(config=config)
        assert chunker.config.parent_chunk_size == 256

    def test_create_parent_chunks(self):
        """Test creating parent chunks."""
        from src.retrieval import HierarchicalChunker

        chunker = HierarchicalChunker()

        # Create 1000 tokens
        tokens = ["word"] * 1000

        parents = chunker._create_parent_chunks(
            "doc1", "Test Doc", tokens
        )

        assert len(parents) == 2  # 512 + 488 tokens
        assert all(chunk.is_parent for chunk in parents)

    def test_create_child_chunks(self):
        """Test creating child chunks from parent."""
        from src.retrieval import HierarchicalChunker

        chunker = HierarchicalChunker()

        tokens = ["word"] * 1000
        parents = chunker._create_parent_chunks(
            "doc1", "Test Doc", tokens
        )

        parent = parents[0]
        children = chunker._create_child_chunks("doc1", parent)

        assert len(children) > 0
        assert all(not chunk.is_parent for chunk in children)
        assert all(chunk.parent_id == parent.chunk_id for chunk in children)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
