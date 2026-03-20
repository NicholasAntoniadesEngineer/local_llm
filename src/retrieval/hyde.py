"""HyDE (Hypothetical Document Embeddings) expansion for query enhancement."""

import asyncio
from typing import Optional
import structlog
import httpx

from ..llm.router import ModelRouter
from ..llm.base import CompletionRequest
from .models import HyDeResult

logger = structlog.get_logger(__name__)


class HyDEExpander:
    """Generate hypothetical documents and average their embeddings with query."""

    def __init__(
        self,
        model_router: ModelRouter,
        num_hypotheticals: int = 3,
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize HyDE expander.

        Args:
            model_router: ModelRouter instance for LLM calls
            num_hypotheticals: Number of hypothetical documents to generate (default 3)
            embedding_model: Name of embedding model
            ollama_base_url: Base URL for Ollama API

        Raises:
            ValueError: If num_hypotheticals < 1
        """
        if num_hypotheticals < 1:
            raise ValueError("num_hypotheticals must be at least 1")

        self.model_router = model_router
        self.num_hypotheticals = num_hypotheticals
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.embedding_dimension = 768  # nomic-embed-text default

    async def expand(self, query: str) -> HyDeResult:
        """
        Expand query with hypothetical documents.

        Process:
        1. Generate N hypothetical documents answering the query
        2. Embed the original query
        3. Embed each hypothetical document
        4. Average all embeddings (query + hypotheticals)

        Args:
            query: The search query

        Returns:
            HyDeResult with query, hypotheticals, embeddings, and averaged result

        Raises:
            ValueError: If query is empty
            RuntimeError: If embedding generation fails
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        logger.info("hyde_expand_start", query=query[:100], num_hypotheticals=self.num_hypotheticals)

        try:
            # Step 1: Generate hypothetical documents
            start_time = asyncio.get_event_loop().time()
            hypotheticals = await self._generate_hypotheticals(query)
            generation_latency = (asyncio.get_event_loop().time() - start_time) * 1000

            # Step 2: Embed query and hypotheticals
            texts_to_embed = [query] + hypotheticals
            embeddings = await self._embed_texts(texts_to_embed)

            if len(embeddings) != len(texts_to_embed):
                raise RuntimeError(
                    f"Expected {len(texts_to_embed)} embeddings, got {len(embeddings)}"
                )

            # Step 3: Separate query and hypothetical embeddings
            query_embedding = embeddings[0]
            hypothetical_embeddings = embeddings[1:]

            # Step 4: Average embeddings
            averaged_embedding = self._average_embeddings(
                [query_embedding] + hypothetical_embeddings
            )

            result = HyDeResult(
                query=query,
                hypotheticals=hypotheticals,
                query_embedding=query_embedding,
                hypothetical_embeddings=hypothetical_embeddings,
                averaged_embedding=averaged_embedding,
                generation_latency_ms=generation_latency,
            )

            logger.info(
                "hyde_expand_complete",
                query=query[:100],
                hypotheticals=len(hypotheticals),
                embedding_dim=len(averaged_embedding),
                latency_ms=generation_latency,
            )

            return result

        except Exception as e:
            logger.error(
                "hyde_expand_failed",
                query=query[:100],
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _generate_hypotheticals(self, query: str) -> list[str]:
        """
        Generate hypothetical documents answering the query.

        Uses the orchestrate role (fast model) to generate 3 hypothetical documents
        in a single call.

        Args:
            query: The search query

        Returns:
            List of hypothetical document texts

        Raises:
            RuntimeError: If generation fails or parsing fails
        """
        system_prompt = """You are an expert at generating realistic documents.
Given a query, generate {num} realistic and diverse documents that would answer this query well.
Each document should be 2-3 sentences long and provide different perspectives or information.
Format your response as a numbered list (1. ..., 2. ..., etc.).""".format(
            num=self.num_hypotheticals
        )

        user_prompt = f"Generate {self.num_hypotheticals} documents answering this query:\n\nQuery: {query}"

        try:
            response = await self.model_router.complete(
                role="orchestrate",
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=800,
            )

            # Parse numbered list from response
            hypotheticals = self._parse_hypotheticals(response.text)

            if len(hypotheticals) < self.num_hypotheticals:
                logger.warning(
                    "hyde_fewer_hypotheticals",
                    expected=self.num_hypotheticals,
                    got=len(hypotheticals),
                    text=response.text[:200],
                )
                # Pad with repeats of last if needed
                while len(hypotheticals) < self.num_hypotheticals:
                    hypotheticals.append(hypotheticals[-1] if hypotheticals else "")

            return hypotheticals[: self.num_hypotheticals]

        except Exception as e:
            logger.error(
                "hyde_generation_failed",
                query=query[:100],
                error=str(e),
            )
            raise RuntimeError(f"Failed to generate hypotheticals: {e}") from e

    def _parse_hypotheticals(self, text: str) -> list[str]:
        """
        Parse numbered list of hypotheticals from LLM response.

        Handles formats like:
        - 1. First document
        - 2. Second document
        - etc.

        Args:
            text: LLM response text

        Returns:
            List of parsed hypothetical documents
        """
        hypotheticals = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match patterns like "1. ", "1) ", "- "
            for prefix in [".", ")", ":"]:
                if len(line) > 2 and line[0].isdigit() and prefix in line:
                    # Find the prefix
                    idx = line.find(prefix)
                    if idx > 0 and idx < 5:  # Reasonable position for number
                        content = line[idx + 1 :].strip()
                        if content:
                            hypotheticals.append(content)
                            break
            else:
                # Also accept lines starting with bullet points
                if line.startswith("-") and len(line) > 1:
                    content = line[1:].strip()
                    if content:
                        hypotheticals.append(content)

        return hypotheticals

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts using Ollama embedding endpoint.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (each 768-dim)

        Raises:
            RuntimeError: If embedding fails
        """
        embeddings = []

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for text in texts:
                    # Use Ollama embedding API
                    response = await client.post(
                        f"{self.ollama_base_url}/api/embeddings",
                        json={
                            "model": self.embedding_model,
                            "prompt": text,
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

                    embeddings.append(embedding)

            logger.info(
                "hyde_embeddings_generated",
                text_count=len(texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0,
            )

            return embeddings

        except Exception as e:
            logger.error(
                "hyde_embedding_failed",
                text_count=len(texts),
                error=str(e),
            )
            raise RuntimeError(f"Failed to embed texts: {e}") from e

    def _average_embeddings(self, embeddings: list[list[float]]) -> list[float]:
        """
        Average multiple embedding vectors element-wise.

        Args:
            embeddings: List of embedding vectors (all same dimension)

        Returns:
            Averaged embedding vector

        Raises:
            ValueError: If embeddings list is empty or dimensions mismatch
        """
        if not embeddings:
            raise ValueError("embeddings list cannot be empty")

        # Verify all have same dimension
        dim = len(embeddings[0])
        for emb in embeddings:
            if len(emb) != dim:
                raise ValueError(f"All embeddings must have same dimension (expected {dim}, got {len(emb)})")

        # Compute element-wise average
        averaged = [sum(emb[i] for emb in embeddings) / len(embeddings) for i in range(dim)]

        return averaged
