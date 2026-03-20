"""Hierarchical chunking with parent-child relationships."""

import asyncio
from typing import Optional
import structlog
import httpx
import re

from .models import Chunk, ChunkerConfig, TokenStats

logger = structlog.get_logger(__name__)


class HierarchicalChunker:
    """
    Chunk documents into parent-child hierarchy.

    Parents: 512 tokens (contextual, for LanceDB storage)
    Children: 128 tokens (retrievable, for search results)
    Semantic boundaries via embedding similarity.

    Parent-doc retrieval pattern:
    - Retrieve child chunks via hybrid search
    - Return parent chunks to LLM for better context
    """

    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ):
        """
        Initialize hierarchical chunker.

        Args:
            config: ChunkerConfig with chunk sizes and strategy
            ollama_base_url: Base URL for Ollama API
            embedding_model: Name of embedding model for semantic boundaries

        Raises:
            ValueError: If config invalid
        """
        self.config = config or ChunkerConfig()
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.embedding_dimension = 768  # nomic-embed-text default

        logger.info(
            "hierarchical_chunker_init",
            config=str(self.config),
            embedding_model=embedding_model,
        )

    async def chunk_document(
        self,
        document_id: str,
        title: str,
        content: str,
        token_counts: Optional[dict[str, int]] = None,
    ) -> tuple[list[Chunk], TokenStats]:
        """
        Chunk a document into parent-child hierarchy.

        Process:
        1. Split content into parent chunks (512 tokens)
        2. For each parent, create child chunks (128 tokens)
        3. Compute embeddings for each chunk
        4. Use semantic similarity to find boundaries
        5. Build parent-child relationships

        Args:
            document_id: Unique document identifier
            title: Document title
            content: Full document content
            token_counts: Optional pre-computed token counts per sentence

        Returns:
            Tuple of (list of Chunk objects, TokenStats)

        Raises:
            ValueError: If document_id or content empty
            RuntimeError: If chunking or embedding fails
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty")

        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        logger.info(
            "chunk_document_start",
            document_id=document_id,
            title=title,
            content_length=len(content),
        )

        try:
            # Step 1: Tokenize content
            tokens = content.split()  # Simple whitespace tokenization
            total_tokens = len(tokens)

            # Step 2: Create parent chunks (512 tokens)
            parent_chunks = self._create_parent_chunks(
                document_id, title, tokens
            )

            # Step 3: Create child chunks from parents (128 tokens)
            all_chunks = []
            for parent in parent_chunks:
                children = self._create_child_chunks(document_id, parent)
                all_chunks.append(parent)
                all_chunks.extend(children)

            # Step 4: Compute embeddings for all chunks
            await self._embed_chunks(all_chunks)

            # Step 5: Optionally refine boundaries via semantic similarity
            # (Can improve chunking by splitting at semantic boundaries)
            # For now, kept simple - semantic refinement is optional enhancement

            logger.info(
                "chunk_document_complete",
                document_id=document_id,
                parent_count=len(parent_chunks),
                total_chunks=len(all_chunks),
                total_tokens=total_tokens,
            )

            # Compute statistics
            stats = TokenStats(
                query_tokens=0,  # Set by caller if relevant
                chunk_tokens=sum(chunk.token_count for chunk in all_chunks),
                total_tokens=total_tokens,
                embedding_cost_tokens=len(all_chunks) * 128,  # Approx embedding cost
            )

            return all_chunks, stats

        except Exception as e:
            logger.error(
                "chunk_document_failed",
                document_id=document_id,
                error=str(e),
            )
            raise

    def _create_parent_chunks(
        self,
        document_id: str,
        title: str,
        tokens: list[str],
    ) -> list[Chunk]:
        """
        Create parent chunks (512 tokens each).

        Args:
            document_id: Document ID
            title: Document title
            tokens: All tokens from document

        Returns:
            List of parent Chunk objects
        """
        parent_chunks = []
        parent_size = self.config.parent_chunk_size

        for i, chunk_idx in enumerate(range(0, len(tokens), parent_size)):
            chunk_tokens = tokens[chunk_idx : chunk_idx + parent_size]

            if len(chunk_tokens) < self.config.min_chunk_size:
                # Skip too-small chunks at end
                if parent_chunks:
                    # Append to last chunk instead
                    parent_chunks[-1].text += " " + " ".join(chunk_tokens)
                    parent_chunks[-1].token_count += len(chunk_tokens)
                continue

            chunk_id = f"{document_id}::{i}"
            chunk_text = " ".join(chunk_tokens)

            parent = Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                position=i,
                is_parent=True,
                metadata={
                    "title": title,
                    "chunk_type": "parent",
                    "chunk_size_tokens": self.config.parent_chunk_size,
                },
            )

            parent_chunks.append(parent)

        logger.info(
            "parent_chunks_created",
            document_id=document_id,
            count=len(parent_chunks),
            parent_size=parent_size,
        )

        return parent_chunks

    def _create_child_chunks(
        self,
        document_id: str,
        parent: Chunk,
    ) -> list[Chunk]:
        """
        Create child chunks (128 tokens) from a parent chunk.

        Args:
            document_id: Document ID
            parent: Parent Chunk

        Returns:
            List of child Chunk objects
        """
        child_chunks = []
        child_size = self.config.child_chunk_size
        parent_tokens = parent.text.split()

        child_idx = 0
        for i in range(0, len(parent_tokens), child_size):
            chunk_tokens = parent_tokens[i : i + child_size]

            if len(chunk_tokens) < self.config.min_chunk_size:
                if child_chunks:
                    # Append to last chunk
                    child_chunks[-1].text += " " + " ".join(chunk_tokens)
                    child_chunks[-1].token_count += len(chunk_tokens)
                continue

            chunk_id = f"{document_id}::{parent.position}_{child_idx}"
            chunk_text = " ".join(chunk_tokens)

            child = Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                position=i // child_size,
                is_parent=False,
                parent_id=parent.chunk_id,
                metadata={
                    "title": parent.metadata.get("title", ""),
                    "chunk_type": "child",
                    "chunk_size_tokens": self.config.child_chunk_size,
                    "parent_id": parent.chunk_id,
                },
            )

            child_chunks.append(child)
            child_idx += 1

        logger.info(
            "child_chunks_created",
            parent_id=parent.chunk_id,
            count=len(child_chunks),
            child_size=child_size,
        )

        return child_chunks

    async def _embed_chunks(self, chunks: list[Chunk]) -> None:
        """
        Embed all chunks using Ollama embedding endpoint.

        Args:
            chunks: List of Chunk objects to embed

        Raises:
            RuntimeError: If embedding fails
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for chunk in chunks:
                    response = await client.post(
                        f"{self.ollama_base_url}/api/embeddings",
                        json={
                            "model": self.embedding_model,
                            "prompt": chunk.text,
                            "normalize": True,  # Use Matryoshka normalized embeddings
                        },
                    )

                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Embedding failed with status {response.status_code}: {response.text}"
                        )

                    data = response.json()
                    embedding = data.get("embedding")
                    if not embedding:
                        raise RuntimeError("No embedding in response")

                    chunk.embedding = embedding

            logger.info(
                "chunks_embedded",
                count=len(chunks),
                embedding_dim=len(chunks[0].embedding) if chunks[0].embedding else 0,
            )

        except Exception as e:
            logger.error(
                "chunk_embedding_failed",
                chunk_count=len(chunks),
                error=str(e),
            )
            raise RuntimeError(f"Failed to embed chunks: {e}") from e

    def refine_boundaries(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """
        Optionally refine chunk boundaries using semantic similarity.

        Finds chunks with low embedding similarity and suggests re-chunking.
        This is an optional enhancement for improved retrieval quality.

        Args:
            chunks: Chunks to refine

        Returns:
            Refined chunks (or original if no improvements found)
        """
        # Optional: Implement semantic boundary detection
        # For now, return chunks as-is
        # Enhancement: Could identify boundaries with low coherence
        # and re-chunk at semantic breaks

        return chunks

    async def batch_chunk_documents(
        self,
        documents: list[dict],
    ) -> dict[str, tuple[list[Chunk], TokenStats]]:
        """
        Chunk multiple documents in parallel.

        Args:
            documents: List of dicts with 'document_id', 'title', 'content'

        Returns:
            Dict mapping document_id → (chunks, stats)

        Raises:
            ValueError: If documents invalid
        """
        if not documents:
            raise ValueError("documents list cannot be empty")

        logger.info("batch_chunk_documents_start", count=len(documents))

        try:
            tasks = [
                self.chunk_document(
                    document["document_id"],
                    document.get("title", ""),
                    document["content"],
                )
                for document in documents
            ]

            results = await asyncio.gather(*tasks)

            output = {
                doc["document_id"]: result
                for doc, result in zip(documents, results)
            }

            total_chunks = sum(
                len(chunks) for chunks, _ in output.values()
            )
            total_tokens = sum(
                stats.total_tokens for _, stats in output.values()
            )

            logger.info(
                "batch_chunk_documents_complete",
                document_count=len(documents),
                total_chunks=total_chunks,
                total_tokens=total_tokens,
            )

            return output

        except Exception as e:
            logger.error(
                "batch_chunk_documents_failed",
                count=len(documents),
                error=str(e),
            )
            raise

    def get_stats(self, chunks: list[Chunk]) -> dict[str, int]:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dict with statistics (count, tokens, etc)
        """
        parent_chunks = [c for c in chunks if c.is_parent]
        child_chunks = [c for c in chunks if not c.is_parent]

        return {
            "total_chunks": len(chunks),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "avg_chunk_tokens": int(sum(c.token_count for c in chunks) / len(chunks)) if chunks else 0,
            "embedded_chunks": sum(1 for c in chunks if c.embedding),
        }
