"""Cross-encoder reranking for search results."""

import asyncio
from typing import Optional
import structlog

from sentence_transformers import CrossEncoder

from .models import SearchResult, RerankerConfig

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """
    Rerank search results using a cross-encoder model.

    Uses ms-marco-MiniLM-L-6-v2: 80MB, CPU-only, suitable for production.
    Reranks top-20 results → top-5 based on query-chunk relevance scores.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize cross-encoder reranker.

        Args:
            config: RerankerConfig with model name, batch size, top_k

        Raises:
            RuntimeError: If model fails to load
        """
        self.config = config or RerankerConfig()
        self.model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load cross-encoder model from HuggingFace.

        Raises:
            RuntimeError: If model load fails
        """
        try:
            logger.info("reranker_loading_model", model=self.config.model_name)

            # Load model on CPU (as specified in config)
            device = "cuda" if self.config.use_gpu else "cpu"
            self.model = CrossEncoder(
                self.config.model_name,
                device=device,
                default_activation_function=None,  # Use raw logits
            )

            logger.info(
                "reranker_model_loaded",
                model=self.config.model_name,
                device=device,
            )

        except Exception as e:
            logger.error(
                "reranker_load_failed",
                model=self.config.model_name,
                error=str(e),
            )
            raise RuntimeError(f"Failed to load cross-encoder: {e}") from e

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Process:
        1. Prepare (query, chunk_text) pairs
        2. Score all pairs in batches
        3. Update results with reranker scores
        4. Sort by reranker score
        5. Return top-k results
        6. Filter by threshold if configured

        Args:
            query: Search query
            results: List of SearchResult to rerank
            top_k: Override default top_k from config

        Returns:
            Top-k reranked SearchResult sorted by score

        Raises:
            ValueError: If query or results empty
            RuntimeError: If reranking fails
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not results:
            logger.info("rerank_empty_results")
            return []

        top_k = top_k or self.config.top_k

        logger.info(
            "rerank_start",
            query=query[:100],
            input_count=len(results),
            top_k=top_k,
            batch_size=self.config.batch_size,
        )

        try:
            if not self.model:
                raise RuntimeError("Model not loaded")

            # Prepare query-chunk pairs
            pairs = [
                [query, result.chunk.text]
                for result in results
            ]

            # Score all pairs in batches
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # Update results with reranker scores
            for result, score in zip(results, scores):
                result.reranker_score = float(score)

            # Sort by reranker score descending
            results.sort(key=lambda x: x.reranker_score or 0.0, reverse=True)

            # Filter by threshold if specified
            if self.config.threshold is not None:
                results = [
                    r for r in results
                    if (r.reranker_score or 0.0) >= self.config.threshold
                ]
                logger.info(
                    "rerank_threshold_filtered",
                    threshold=self.config.threshold,
                    results_after=len(results),
                )

            # Return top-k
            top_results = results[:top_k]

            logger.info(
                "rerank_complete",
                query=query[:100],
                output_count=len(top_results),
                top_score=top_results[0].reranker_score if top_results else None,
            )

            return top_results

        except Exception as e:
            logger.error(
                "rerank_failed",
                query=query[:100],
                error=str(e),
            )
            raise RuntimeError(f"Reranking failed: {e}") from e

    async def rerank_async(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Async wrapper for reranking.

        Runs reranking in thread pool to avoid blocking.

        Args:
            query: Search query
            results: Results to rerank
            top_k: Override top_k from config

        Returns:
            Top-k reranked results

        Raises:
            ValueError: If query or results empty
            RuntimeError: If reranking fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rerank,
            query,
            results,
            top_k,
        )

    def get_model_info(self) -> dict[str, str]:
        """
        Get information about loaded model.

        Returns:
            Dict with model name, device, and parameters
        """
        if not self.model:
            return {"status": "not_loaded"}

        return {
            "model_name": self.config.model_name,
            "max_length": str(self.config.input_max_length),
            "batch_size": str(self.config.batch_size),
            "top_k": str(self.config.top_k),
        }
