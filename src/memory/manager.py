"""Unified memory coordinator managing three-tier memory system."""

from typing import Any, Optional, Literal
from datetime import datetime
import asyncio
import structlog

from .models import (
    SessionInfo,
    ConversationMessage,
    Finding,
    Checkpoint,
    RuleUpdate,
    ActionLog,
    ContextBudget,
)
from .working import WorkingMemory
from .sqlite_store import SQLiteStore
from .lancedb_store import LanceDBStore

logger = structlog.get_logger(__name__)


class MemoryManager:
    """
    Unified memory coordinator for three-tier memory system.

    Tiers:
    - Tier 1: Working memory (in-context deque, 4K tokens)
    - Tier 2: Semantic memory (LanceDB, hybrid search, infinite)
    - Tier 3: Metadata + checkpoints (SQLite, structured data)
    """

    def __init__(
        self,
        sqlite_path: str = "data/agent_state.db",
        lancedb_path: str = "data/research_memory.lancedb",
        working_memory_tokens: int = 4096,
    ):
        """
        Initialize memory manager.

        Args:
            sqlite_path: Path to SQLite database
            lancedb_path: Path to LanceDB directory
            working_memory_tokens: Max tokens for working memory
        """
        self.sqlite_path = sqlite_path
        self.lancedb_path = lancedb_path

        self.working = WorkingMemory(max_tokens=working_memory_tokens)
        self.episodic = SQLiteStore(db_path=sqlite_path)
        self.semantic = LanceDBStore(db_path=lancedb_path)

        self._context_budget = ContextBudget()
        self._current_session: Optional[SessionInfo] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize all memory stores."""
        await asyncio.gather(
            self.episodic.initialize(),
            self.semantic.initialize(),
        )
        logger.info("memory_manager_initialized")

    async def start_session(self, objective: str, max_steps: int = 10) -> SessionInfo:
        """
        Start a new research session.

        Args:
            objective: Research objective/goal
            max_steps: Maximum steps before completion

        Returns:
            Created SessionInfo
        """
        async with self._lock:
            session = SessionInfo(
                objective=objective,
                max_steps=max_steps,
                status="active",
            )

            await self.episodic.save_session(session)
            self._current_session = session

            logger.info("session_started", session_id=session.id, objective=objective[:50])
            return session

    async def resume_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Resume an existing session from checkpoint.

        Args:
            session_id: Session ID to resume

        Returns:
            Resumed SessionInfo or None if not found
        """
        async with self._lock:
            session = await self.episodic.get_session(session_id)

            if session:
                # Load latest checkpoint
                checkpoint = await self.episodic.get_latest_checkpoint(session_id)
                self._current_session = session
                logger.info("session_resumed", session_id=session_id)
                return session

            logger.warning("session_not_found", session_id=session_id)
            return None

    async def end_session(self, status: Literal["completed", "failed", "paused"] = "completed") -> None:
        """
        End current session.

        Args:
            status: Final status (completed, failed, paused)
        """
        async with self._lock:
            if self._current_session:
                self._current_session.status = status
                self._current_session.updated_at = datetime.utcnow()
                await self.episodic.save_session(self._current_session)

                logger.info("session_ended", session_id=self._current_session.id, status=status)
                self._current_session = None

    def get_current_session(self) -> Optional[SessionInfo]:
        """Get current active session."""
        return self._current_session

    # ==================== Conversation Management ====================

    async def add_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
        tokens: int,
        model_used: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        latency_ms: float = 0.0,
    ) -> ConversationMessage:
        """
        Add a conversation message to all tiers.

        Args:
            role: Message role
            content: Message content
            tokens: Token count
            model_used: Model that generated this (if assistant)
            tool_calls: Tool calls in message (if any)
            latency_ms: Response latency

        Returns:
            Created ConversationMessage
        """
        if not self._current_session:
            raise ValueError("No active session")

        # Create message
        message = ConversationMessage(
            session_id=self._current_session.id,
            turn_number=self._current_session.messages_count,
            role=role,
            content=content,
            tokens=tokens,
            model_used=model_used,
            tool_calls=tool_calls or [],
            latency_ms=latency_ms,
        )

        # Save to all tiers
        await asyncio.gather(
            self.episodic.save_conversation(message),
            self.semantic.insert_entry(
                entry_id=message.id,
                text=content,
                vector=[0.0] * 768,  # Placeholder - should be embedded by caller
                entry_type="conversation",
                session_id=self._current_session.id,
                importance=5,
            ),
        )

        # Add to working memory
        self.working.add_message(message)

        # Update session
        self._current_session.messages_count += 1
        self._current_session.total_tokens += tokens
        self._current_session.total_latency_ms += latency_ms
        await self.episodic.save_session(self._current_session)

        logger.debug(
            "message_added",
            message_id=message.id,
            role=role,
            tokens=tokens,
            session_id=self._current_session.id,
        )

        return message

    async def get_conversation_history(
        self, limit: int = 50, from_working_only: bool = False
    ) -> list[ConversationMessage]:
        """
        Get conversation history.

        Args:
            limit: Max messages to return
            from_working_only: If True, only return working memory

        Returns:
            List of ConversationMessage objects
        """
        if from_working_only:
            return self.working.get_buffer()

        if not self._current_session:
            return []

        return await self.episodic.get_conversation_history(
            self._current_session.id, limit=limit
        )

    # ==================== Finding Management ====================

    async def add_finding(
        self,
        title: str,
        content: str,
        sources: list[str] | None = None,
        confidence: float = 0.5,
        tags: list[str] | None = None,
        importance: int = 5,
        category: str = "general",
        raw_text: str | None = None,
    ) -> Finding:
        """
        Add a research finding.

        Args:
            title: Finding title
            content: Finding summary
            sources: Source references
            confidence: Confidence score (0-1)
            tags: Categorization tags
            importance: Importance score (1-10)
            category: Category classification
            raw_text: Raw extracted text

        Returns:
            Created Finding
        """
        if not self._current_session:
            raise ValueError("No active session")

        finding = Finding(
            session_id=self._current_session.id,
            title=title,
            content=content,
            sources=sources or [],
            confidence=confidence,
            tags=tags or [],
            importance=importance,
            category=category,
            raw_text=raw_text,
        )

        # Save to all tiers
        await asyncio.gather(
            self.episodic.save_finding(finding),
            self.semantic.insert_entry(
                entry_id=finding.id,
                text=f"{title}: {content}",
                vector=[0.0] * 768,  # Placeholder
                entry_type="finding",
                session_id=self._current_session.id,
                importance=importance,
                tags=tags,
                source=sources[0] if sources else None,
                metadata={"category": category, "confidence": confidence},
            ),
        )

        # Update session
        self._current_session.findings_count += 1
        await self.episodic.save_session(self._current_session)

        logger.debug(
            "finding_added",
            finding_id=finding.id,
            session_id=self._current_session.id,
            confidence=confidence,
        )

        return finding

    async def get_findings(
        self, limit: int = 50, min_confidence: float = 0.0
    ) -> list[Finding]:
        """
        Get findings for current session.

        Args:
            limit: Max findings to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of Finding objects
        """
        if not self._current_session:
            return []

        return await self.episodic.get_findings(
            self._current_session.id, limit=limit, min_confidence=min_confidence
        )

    # ==================== Checkpoint Management ====================

    async def save_checkpoint(
        self,
        agent_state: dict[str, Any],
        memory_state: dict[str, Any],
        completed_actions: list[str] | None = None,
        next_actions: list[str] | None = None,
    ) -> Checkpoint:
        """
        Save a checkpoint for resumability.

        Args:
            agent_state: Agent execution state
            memory_state: Memory system state
            completed_actions: List of completed action IDs
            next_actions: List of planned action IDs

        Returns:
            Created Checkpoint
        """
        if not self._current_session:
            raise ValueError("No active session")

        checkpoint = Checkpoint(
            session_id=self._current_session.id,
            step_number=self._current_session.current_step,
            agent_state=agent_state,
            memory_state=memory_state,
            completed_actions=completed_actions or [],
            next_actions=next_actions or [],
        )

        await self.episodic.save_checkpoint(checkpoint)

        logger.debug(
            "checkpoint_saved",
            checkpoint_id=checkpoint.id,
            step=self._current_session.current_step,
        )

        return checkpoint

    async def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get latest checkpoint for current session."""
        if not self._current_session:
            return None

        return await self.episodic.get_latest_checkpoint(self._current_session.id)

    # ==================== Rule Management ====================

    async def log_rule_update(
        self,
        rule_id: str,
        rule_type: Literal["hard", "soft"],
        new_rule: str,
        reason: str,
        old_rule: str | None = None,
    ) -> RuleUpdate:
        """
        Log a proposed rule update.

        Args:
            rule_id: Rule identifier
            rule_type: Type of rule (hard/soft)
            new_rule: New rule text
            reason: Reason for update
            old_rule: Previous rule text (if updating)

        Returns:
            Created RuleUpdate
        """
        update = RuleUpdate(
            rule_id=rule_id,
            rule_type=rule_type,
            old_rule=old_rule,
            new_rule=new_rule,
            reason=reason,
            status="proposed",
        )

        await self.episodic.save_rule_update(update)

        logger.debug("rule_update_logged", rule_update_id=update.id, rule_id=rule_id)

        return update

    # ==================== Action Logging ====================

    async def log_action(
        self,
        action_type: str,
        tool_name: str | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        status: Literal["pending", "success", "failed"] = "pending",
        error_message: str | None = None,
        duration_ms: float = 0.0,
    ) -> ActionLog:
        """
        Log a tool or action execution.

        Args:
            action_type: Type of action
            tool_name: Name of tool (if applicable)
            input_data: Input parameters
            output_data: Output/results
            status: Execution status
            error_message: Error message if failed
            duration_ms: Execution time in milliseconds

        Returns:
            Created ActionLog
        """
        if not self._current_session:
            raise ValueError("No active session")

        action = ActionLog(
            session_id=self._current_session.id,
            action_type=action_type,
            tool_name=tool_name,
            input_data=input_data or {},
            output_data=output_data or {},
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
        )

        await self.episodic.save_action(action)

        # Also log to semantic for searchability
        await self.semantic.insert_entry(
            entry_id=action.id,
            text=f"{action_type}: {tool_name or 'unknown'}",
            vector=[0.0] * 768,
            entry_type="action",
            session_id=self._current_session.id,
            metadata={
                "status": status,
                "tool": tool_name,
                "duration_ms": duration_ms,
            },
        )

        logger.debug(
            "action_logged",
            action_id=action.id,
            action_type=action_type,
            status=status,
        )

        return action

    async def get_actions(
        self, limit: int = 50, status: str | None = None
    ) -> list[ActionLog]:
        """
        Get action logs for current session.

        Args:
            limit: Max actions to return
            status: Filter by status (optional)

        Returns:
            List of ActionLog objects
        """
        if not self._current_session:
            return []

        return await self.episodic.get_actions(
            self._current_session.id, limit=limit, status=status
        )

    # ==================== Memory Retrieval ====================

    async def search_memory(
        self,
        query: str,
        query_vector: list[float] | None = None,
        limit: int = 10,
        search_type: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> list[dict[str, Any]]:
        """
        Search memory system.

        Args:
            query: Search query text
            query_vector: Optional vector for similarity search
            limit: Max results
            search_type: Type of search to perform

        Returns:
            List of search results
        """
        if not self._current_session:
            return []

        query_vector = query_vector or [0.0] * 768

        if search_type == "hybrid":
            return await self.semantic.hybrid_search(
                query=query,
                query_vector=query_vector,
                session_id=self._current_session.id,
                limit=limit,
            )
        elif search_type == "vector":
            return await self.semantic.vector_search(
                query_vector=query_vector,
                session_id=self._current_session.id,
                limit=limit,
            )
        else:  # bm25
            return await self.semantic.bm25_search(
                query=query,
                session_id=self._current_session.id,
                limit=limit,
            )

    async def get_important_memories(
        self, min_importance: int = 7, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get high-importance memories.

        Args:
            min_importance: Minimum importance threshold
            limit: Max results

        Returns:
            List of important memory entries
        """
        if not self._current_session:
            return []

        return await self.semantic.get_important_entries(
            self._current_session.id, min_importance=min_importance, limit=limit
        )

    # ==================== Context Management ====================

    def get_context_budget(self) -> ContextBudget:
        """Get context budget allocation."""
        return self._context_budget

    def get_working_memory_context(self) -> dict[str, Any]:
        """Get formatted context from working memory."""
        return self.working.get_context_for_model(self._context_budget)

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "working": self.working.stats(),
            "session": None,
            "context_budget": {
                "system_prompt": self._context_budget.system_prompt,
                "tool_definitions": self._context_budget.tool_definitions,
                "retrieved_memory": self._context_budget.retrieved_memory,
                "conversation_history": self._context_budget.conversation_history,
                "workspace_scratch": self._context_budget.workspace_scratch,
                "response_buffer": self._context_budget.response_buffer,
                "total": self._context_budget.total_budget,
            },
        }

        if self._current_session:
            stats["session"] = {
                "id": self._current_session.id,
                "objective": self._current_session.objective,
                "status": self._current_session.status,
                "messages": self._current_session.messages_count,
                "findings": self._current_session.findings_count,
                "total_tokens": self._current_session.total_tokens,
            }

        return stats

    # ==================== Cleanup ====================

    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Delete old sessions to free storage.

        Args:
            days_old: Delete sessions older than this many days

        Returns:
            Number of sessions deleted
        """
        count = await self.episodic.cleanup_old_sessions(days_old=days_old)
        logger.info("cleanup_completed", sessions_deleted=count, days_old=days_old)
        return count

    async def close(self) -> None:
        """Close all memory stores."""
        await asyncio.gather(
            self.episodic.close(),
            self.semantic.close(),
        )
        logger.info("memory_manager_closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()
