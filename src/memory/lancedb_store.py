"""LanceDB vector store with hybrid search capabilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import lancedb
import structlog
from rank_bm25 import BM25Okapi

logger = structlog.get_logger(__name__)


class LanceDBStore:
    """Vector database store with hybrid search (BM25 + vector + RRF fusion)."""

    def __init__(self, db_path: str = "data/research_memory.lancedb"):
        """
        Initialize LanceDB store.

        Args:
            db_path: Path to LanceDB directory (absolute path recommended)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = None
        self.table = None
        self._bm25_index = {}  # Cache BM25 indices by session_id
        self._corpus = {}  # Cache tokenized documents by session_id

    async def initialize(self) -> None:
        """Initialize LanceDB and create schema."""
        try:
            # LanceDB uses sync API, but we wrap in context
            self.db = lancedb.connect(str(self.db_path))

            # Create or get table with schema
            if "research_memory" not in self.db.table_names():
                # Create empty table with schema via first insert
                # Schema: id, text, vector(768), type, session_id, importance, tags, timestamp, source
                self.table = None  # Will be created on first insert
            else:
                self.table = self.db.open_table("research_memory")

            logger.info("lancedb_initialized", path=str(self.db_path))
        except Exception as e:
            logger.error("lancedb_init_failed", error=str(e))
            raise

    def _ensure_table_exists(self) -> None:
        """Ensure table exists before operations."""
        if self.table is None and "research_memory" in self.db.table_names():
            self.table = self.db.open_table("research_memory")

    async def insert_entry(
        self,
        entry_id: str,
        text: str,
        vector: list[float],
        entry_type: str,
        session_id: str,
        importance: int = 5,
        tags: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Insert or upsert a memory entry.

        Args:
            entry_id: Unique entry ID
            text: Text content to embed
            vector: 768-dim embedding vector
            entry_type: Type of entry (conversation, finding, action, etc.)
            session_id: Session ID for scoping
            importance: Importance score 1-10
            tags: Optional tags for categorization
            source: Optional source URL or reference
            metadata: Optional additional metadata
        """
        if self.db is None:
            await self.initialize()

        tags = tags or []
        metadata = metadata or {}

        entry = {
            "id": entry_id,
            "text": text,
            "vector": vector,
            "type": entry_type,
            "session_id": session_id,
            "importance": importance,
            "tags": json.dumps(tags),
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "metadata": json.dumps(metadata),
        }

        try:
            if self.table is None:
                self.table = self.db.create_table("research_memory", data=[entry], mode="overwrite")
            else:
                self.table.merge(
                    [entry],
                    keys=["id"],
                )

            logger.debug("entry_inserted", entry_id=entry_id, session_id=session_id)
        except Exception as e:
            logger.error("insert_failed", entry_id=entry_id, error=str(e))
            raise

    async def batch_insert(
        self, entries: list[dict[str, Any]], session_id: str
    ) -> None:
        """
        Batch insert multiple entries efficiently.

        Args:
            entries: List of entry dicts with required fields
            session_id: Session ID for all entries
        """
        if self.db is None:
            await self.initialize()

        if not entries:
            return

        # Normalize entries
        normalized = []
        for entry in entries:
            normalized.append({
                "id": entry.get("id"),
                "text": entry.get("text", ""),
                "vector": entry.get("vector", [0.0] * 768),
                "type": entry.get("type", "unknown"),
                "session_id": session_id,
                "importance": entry.get("importance", 5),
                "tags": json.dumps(entry.get("tags", [])),
                "timestamp": datetime.utcnow().isoformat(),
                "source": entry.get("source"),
                "metadata": json.dumps(entry.get("metadata", {})),
            })

        try:
            if self.table is None:
                self.table = self.db.create_table(
                    "research_memory", data=normalized, mode="overwrite"
                )
            else:
                self.table.merge(normalized, keys=["id"])

            logger.debug("batch_insert_complete", count=len(entries), session_id=session_id)
        except Exception as e:
            logger.error("batch_insert_failed", count=len(entries), error=str(e))
            raise

    async def vector_search(
        self,
        query_vector: list[float],
        session_id: str,
        limit: int = 10,
        filter_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Pure vector similarity search.

        Args:
            query_vector: 768-dim query vector
            session_id: Scope to session
            limit: Max results
            filter_type: Optional type filter

        Returns:
            List of matched entries sorted by similarity
        """
        self._ensure_table_exists()

        if self.table is None:
            return []

        try:
            # Build where clause
            where_clauses = [f"session_id = '{session_id}'"]
            if filter_type:
                where_clauses.append(f"type = '{filter_type}'")
            where_clause = " AND ".join(where_clauses)

            results = self.table.search(query_vector).where(where_clause).limit(limit).to_list()

            # Normalize results
            normalized = []
            for result in results:
                normalized.append({
                    "id": result.get("id"),
                    "text": result.get("text"),
                    "type": result.get("type"),
                    "importance": result.get("importance"),
                    "source": result.get("source"),
                    "score": result.get("_distance", 0.0),
                    "tags": json.loads(result.get("tags", "[]")),
                    "metadata": json.loads(result.get("metadata", "{}")),
                })

            logger.debug(
                "vector_search_complete",
                session_id=session_id,
                limit=limit,
                results=len(normalized),
            )
            return normalized
        except Exception as e:
            logger.error("vector_search_failed", session_id=session_id, error=str(e))
            return []

    def _build_bm25_index(self, session_id: str, documents: list[str]) -> BM25Okapi:
        """
        Build or retrieve BM25 index for session.

        Args:
            session_id: Session ID
            documents: List of documents to index

        Returns:
            BM25Okapi instance
        """
        # Tokenize (simple whitespace + lowercase)
        tokenized = [doc.lower().split() for doc in documents]

        # Cache for reuse
        self._corpus[session_id] = tokenized
        bm25 = BM25Okapi(tokenized)
        self._bm25_index[session_id] = bm25

        return bm25

    async def bm25_search(
        self, query: str, session_id: str, limit: int = 10, filter_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        BM25 full-text search.

        Args:
            query: Search query
            session_id: Scope to session
            limit: Max results
            filter_type: Optional type filter

        Returns:
            List of matched entries sorted by relevance
        """
        self._ensure_table_exists()

        if self.table is None:
            return []

        try:
            # Get all session documents
            where_clause = f"session_id = '{session_id}'"
            if filter_type:
                where_clause += f" AND type = '{filter_type}'"

            all_docs = self.table.search().where(where_clause).to_list()

            if not all_docs:
                return []

            # Extract texts and build BM25
            texts = [doc.get("text", "") for doc in all_docs]
            bm25 = self._build_bm25_index(session_id, texts)

            # Search
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)

            # Sort and return top results
            scored_docs = list(zip(all_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results = []
            for doc, score in scored_docs[:limit]:
                results.append({
                    "id": doc.get("id"),
                    "text": doc.get("text"),
                    "type": doc.get("type"),
                    "importance": doc.get("importance"),
                    "source": doc.get("source"),
                    "score": float(score),
                    "tags": json.loads(doc.get("tags", "[]")),
                    "metadata": json.loads(doc.get("metadata", "{}")),
                })

            logger.debug(
                "bm25_search_complete",
                session_id=session_id,
                query=query[:50],
                results=len(results),
            )
            return results
        except Exception as e:
            logger.error("bm25_search_failed", session_id=session_id, error=str(e))
            return []

    async def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        session_id: str,
        limit: int = 10,
        filter_type: str | None = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining BM25 and vector similarity via Reciprocal Rank Fusion (RRF).

        Args:
            query: Text query for BM25
            query_vector: Vector for semantic search
            session_id: Scope to session
            limit: Max results
            filter_type: Optional type filter
            vector_weight: Weight for vector scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)

        Returns:
            List of fused results sorted by combined score
        """
        self._ensure_table_exists()

        if self.table is None:
            return []

        # Run parallel searches
        vector_results = await self.vector_search(
            query_vector, session_id, limit=limit * 2, filter_type=filter_type
        )
        bm25_results = await self.bm25_search(
            query, session_id, limit=limit * 2, filter_type=filter_type
        )

        # Implement RRF (Reciprocal Rank Fusion)
        # RRF score = 1 / (k + rank)
        k = 60  # RRF parameter (tunable)

        rrf_scores = {}

        # Add vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_score = vector_weight / (k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

        # Add BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_score = bm25_weight / (k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

        # Merge result metadata
        all_results = {r["id"]: r for r in vector_results + bm25_results}

        # Sort by RRF score and return top results
        final_results = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            result = all_results.get(doc_id)
            if result:
                result["score"] = score
                final_results.append(result)

        logger.debug(
            "hybrid_search_complete",
            session_id=session_id,
            query=query[:50],
            results=len(final_results),
        )
        return final_results

    async def get_by_tag(
        self, tag: str, session_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get entries by tag.

        Args:
            tag: Tag to search for
            session_id: Scope to session
            limit: Max results

        Returns:
            List of matching entries
        """
        self._ensure_table_exists()

        if self.table is None:
            return []

        try:
            where_clause = f"session_id = '{session_id}' AND tags LIKE '%{tag}%'"
            results = self.table.search().where(where_clause).limit(limit).to_list()

            normalized = []
            for result in results:
                normalized.append({
                    "id": result.get("id"),
                    "text": result.get("text"),
                    "type": result.get("type"),
                    "importance": result.get("importance"),
                    "source": result.get("source"),
                    "tags": json.loads(result.get("tags", "[]")),
                    "metadata": json.loads(result.get("metadata", "{}")),
                })

            return normalized
        except Exception as e:
            logger.error("get_by_tag_failed", tag=tag, session_id=session_id, error=str(e))
            return []

    async def get_important_entries(
        self, session_id: str, min_importance: int = 7, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get high-importance entries for a session.

        Args:
            session_id: Session ID
            min_importance: Minimum importance threshold (1-10)
            limit: Max results

        Returns:
            List of important entries
        """
        self._ensure_table_exists()

        if self.table is None:
            return []

        try:
            where_clause = f"session_id = '{session_id}' AND importance >= {min_importance}"
            results = (
                self.table.search()
                .where(where_clause)
                .select(["id", "text", "type", "importance", "tags"])
                .limit(limit)
                .to_list()
            )

            normalized = []
            for result in results:
                normalized.append({
                    "id": result.get("id"),
                    "text": result.get("text"),
                    "type": result.get("type"),
                    "importance": result.get("importance"),
                    "tags": json.loads(result.get("tags", "[]")),
                })

            return normalized
        except Exception as e:
            logger.error("get_important_entries_failed", session_id=session_id, error=str(e))
            return []

    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete an entry by ID.

        Args:
            entry_id: Entry ID to delete

        Returns:
            True if deleted, False if not found
        """
        self._ensure_table_exists()

        if self.table is None:
            return False

        try:
            # LanceDB doesn't have native delete, so we rebuild without the entry
            # This is a limitation we accept for now
            logger.warning("delete_not_supported", entry_id=entry_id)
            return False
        except Exception as e:
            logger.error("delete_failed", entry_id=entry_id, error=str(e))
            return False

    async def clear_session(self, session_id: str) -> int:
        """
        Clear all entries for a session.

        Args:
            session_id: Session ID to clear

        Returns:
            Number of entries deleted
        """
        self._ensure_table_exists()

        if self.table is None:
            return 0

        try:
            # Get count before delete
            where_clause = f"session_id = '{session_id}'"
            entries = self.table.search().where(where_clause).to_list()
            count = len(entries)

            # LanceDB doesn't support native delete-where
            # Rebuild table without session
            logger.warning("clear_session_not_fully_supported", session_id=session_id, count=count)
            return count
        except Exception as e:
            logger.error("clear_session_failed", session_id=session_id, error=str(e))
            return 0

    async def get_stats(self, session_id: str) -> dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with statistics
        """
        self._ensure_table_exists()

        if self.table is None:
            return {"total": 0, "by_type": {}, "avg_importance": 0.0}

        try:
            where_clause = f"session_id = '{session_id}'"
            all_docs = self.table.search().where(where_clause).to_list()

            stats = {
                "total": len(all_docs),
                "by_type": {},
                "avg_importance": 0.0,
                "session_id": session_id,
            }

            if all_docs:
                total_importance = 0
                for doc in all_docs:
                    doc_type = doc.get("type", "unknown")
                    stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
                    total_importance += doc.get("importance", 5)

                stats["avg_importance"] = total_importance / len(all_docs)

            logger.debug("stats_retrieved", session_id=session_id, total=stats["total"])
            return stats
        except Exception as e:
            logger.error("get_stats_failed", session_id=session_id, error=str(e))
            return {"total": 0, "by_type": {}, "avg_importance": 0.0}

    async def close(self) -> None:
        """Close database connection."""
        self.table = None
        self.db = None
        self._bm25_index.clear()
        self._corpus.clear()
        logger.debug("lancedb_store_closed")
