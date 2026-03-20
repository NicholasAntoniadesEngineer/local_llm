"""Hybrid search: BM25 + Vector search with RRF fusion."""

import asyncio
from typing import Optional
import structlog
from collections import defaultdict

from rank_bm25 import BM25Okapi

from .models import Chunk, SearchResult, HybridSearchConfig

logger = structlog.get_logger(__name__)


class HybridSearcher:
    """Combines BM25 and vector search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        chunks: Optional[list[Chunk]] = None,
        config: Optional[HybridSearchConfig] = None,
    ):
        """
        Initialize hybrid searcher.

        Args:
            chunks: Optional list of chunks to index (can add later)
            config: HybridSearchConfig with weights and parameters

        Raises:
            ValueError: If chunks provided but config invalid
        """
        self.config = config or HybridSearchConfig()
        self.chunks = chunks or []
        self.chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25_index: Optional[BM25Okapi] = None

        if self.chunks:
            self._build_bm25_index()

        logger.info(
            "hybrid_searcher_init",
            chunk_count=len(self.chunks),
            config=str(self.config),
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add chunks to the index.

        Args:
            chunks: Chunks to add

        Raises:
            ValueError: If chunk IDs not unique
        """
        # Check for duplicates
        existing_ids = set(self.chunk_map.keys())
        new_ids = {chunk.chunk_id for chunk in chunks}
        duplicates = existing_ids & new_ids

        if duplicates:
            raise ValueError(f"Duplicate chunk IDs: {duplicates}")

        self.chunks.extend(chunks)
        self.chunk_map.update({chunk.chunk_id: chunk for chunk in chunks})
        self._build_bm25_index()

        logger.info(
            "hybrid_searcher_chunks_added",
            count=len(chunks),
            total_chunks=len(self.chunks),
        )

    def _build_bm25_index(self) -> None:
        """Build BM25 index from chunks."""
        if not self.chunks:
            self.bm25_index = None
            return

        # Tokenize all chunks
        tokenized_corpus = [
            chunk.text.lower().split() for chunk in self.chunks
        ]

        # Build BM25 index with configured parameters
        self.bm25_index = BM25Okapi(
            tokenized_corpus,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

        logger.info(
            "bm25_index_built",
            chunk_count=len(self.chunks),
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: combine BM25 and vector rankings via RRF.

        Algorithm:
        1. BM25 ranking of all chunks
        2. Vector search ranking (from pre-computed embeddings)
        3. RRF fusion: score = α/(K+bm25_rank) + (1-α)/(K+vector_rank)
        4. Deduplication and reranking
        5. Top-k results

        Args:
            query: Search query text
            query_embedding: Query embedding vector (768-dim)
            top_k: Override default top_k from config

        Returns:
            List of SearchResult ranked by combined score

        Raises:
            ValueError: If query empty or embeddings missing
            RuntimeError: If no chunks indexed
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not self.chunks:
            raise RuntimeError("No chunks indexed")

        top_k = top_k or self.config.top_k

        logger.info(
            "hybrid_search_start",
            query=query[:100],
            top_k=top_k,
            alpha=self.config.alpha,
            total_chunks=len(self.chunks),
        )

        try:
            # Step 1: BM25 search
            bm25_results = self._bm25_search(query)

            # Step 2: Vector search
            vector_results = self._vector_search(query_embedding)

            # Step 3: Combine rankings via RRF
            combined_results = self._rrf_fusion(
                bm25_results, vector_results, query
            )

            # Step 4: Deduplication if enabled
            if self.config.deduplicate:
                combined_results = self._deduplicate(combined_results)

            # Step 5: Normalize scores if enabled
            if self.config.normalize_scores:
                combined_results = self._normalize_scores(combined_results)

            # Step 6: Return top-k
            top_results = combined_results[:top_k]

            # Assign ranks
            for rank, result in enumerate(top_results, 1):
                result.rank = rank

            logger.info(
                "hybrid_search_complete",
                query=query[:100],
                results_count=len(top_results),
                top_score=top_results[0].score if top_results else 0.0,
            )

            return top_results

        except Exception as e:
            logger.error(
                "hybrid_search_failed",
                query=query[:100],
                error=str(e),
            )
            raise

    def _bm25_search(self, query: str) -> list[tuple[str, float]]:
        """
        BM25 search across chunks.

        Args:
            query: Search query

        Returns:
            List of (chunk_id, bm25_score) tuples sorted by score
        """
        if not self.bm25_index:
            return []

        # Tokenize query
        tokens = query.lower().split()

        # Score all chunks
        scores = self.bm25_index.get_scores(tokens)

        # Create results with chunk IDs and scores
        results = [
            (self.chunks[i].chunk_id, float(score))
            for i, score in enumerate(scores)
        ]

        # Sort by score descending, filter zero scores
        results = [(cid, score) for cid, score in results if score > 0.0]
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "bm25_search_done",
            query=query[:50],
            matched_chunks=len(results),
        )

        return results

    def _vector_search(self, query_embedding: list[float]) -> list[tuple[str, float]]:
        """
        Vector search using cosine similarity.

        Args:
            query_embedding: Query embedding vector (768-dim)

        Returns:
            List of (chunk_id, cosine_similarity) tuples sorted by score
        """
        if not self.chunks:
            return []

        results = []

        for chunk in self.chunks:
            if not chunk.embedding:
                continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(
                query_embedding, chunk.embedding
            )
            results.append((chunk.chunk_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "vector_search_done",
            matched_chunks=len(results),
            top_similarity=results[0][1] if results else 0.0,
        )

        return results

    def _cosine_similarity(
        self, vec1: list[float], vec2: list[float]
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1 for normalized vectors)

        Raises:
            ValueError: If vectors have different dimensions
        """
        if len(vec1) != len(vec2):
            raise ValueError(
                f"Vector dimensions mismatch: {len(vec1)} vs {len(vec2)}"
            )

        # For normalized embeddings, cosine similarity = dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return dot_product

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
        query: str,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion: combine BM25 and vector rankings.

        Formula: score = α/(K+bm25_rank) + (1-α)/(K+vector_rank)
        - α=0.5: equal weight to BM25 and vector
        - α→1.0: prefer BM25 (keyword matching)
        - α→0.0: prefer vector (semantic matching)

        Args:
            bm25_results: BM25 ranking (chunk_id, score)
            vector_results: Vector ranking (chunk_id, similarity)
            query: Original query (for logging)

        Returns:
            List of SearchResult sorted by combined score
        """
        # Create rank maps
        bm25_ranks = {chunk_id: i for i, (chunk_id, _) in enumerate(bm25_results, 1)}
        vector_ranks = {chunk_id: i for i, (chunk_id, _) in enumerate(vector_results, 1)}

        # All chunks that appeared in either ranking
        all_chunk_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        K = self.config.rrf_constant
        alpha = self.config.alpha

        results = []

        for chunk_id in all_chunk_ids:
            bm25_rank = bm25_ranks.get(chunk_id, len(bm25_ranks) + 1)
            vector_rank = vector_ranks.get(chunk_id, len(vector_ranks) + 1)

            # RRF formula
            combined_score = (
                alpha / (K + bm25_rank) +
                (1.0 - alpha) / (K + vector_rank)
            )

            chunk = self.chunk_map[chunk_id]
            result = SearchResult(
                chunk=chunk,
                score=combined_score,
                bm25_rank=bm25_rank if chunk_id in bm25_ranks else None,
                vector_rank=vector_rank if chunk_id in vector_ranks else None,
                retrieval_method="hybrid",
            )
            results.append(result)

        # Sort by combined score descending
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "rrf_fusion_done",
            query=query[:50],
            total_results=len(results),
            alpha=alpha,
            top_score=results[0].score if results else 0.0,
        )

        return results

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove duplicate documents, keeping highest scoring occurrence.

        Args:
            results: Ranked search results

        Returns:
            Deduplicated results
        """
        seen_docs = {}
        deduped = []

        for result in results:
            doc_id = result.chunk.document_id

            if doc_id not in seen_docs:
                seen_docs[doc_id] = result
                deduped.append(result)
            # else: skip duplicate (we already have higher-scoring one)

        logger.info(
            "deduplication_done",
            original_count=len(results),
            deduplicated_count=len(deduped),
        )

        return deduped

    def _normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Normalize scores to 0-1 range.

        Uses min-max normalization: (score - min) / (max - min)

        Args:
            results: Search results with scores

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores equal, set to 1.0
            for result in results:
                result.score = 1.0
        else:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)

        logger.info(
            "score_normalization_done",
            result_count=len(results),
            normalized_range=[min(r.score for r in results), max(r.score for r in results)],
        )

        return results

    def clear(self) -> None:
        """Clear all indexed chunks."""
        self.chunks.clear()
        self.chunk_map.clear()
        self.bm25_index = None
        logger.info("hybrid_searcher_cleared")
