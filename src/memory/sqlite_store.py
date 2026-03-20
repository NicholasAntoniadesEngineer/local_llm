"""SQLite async storage for metadata, checkpoints, and structured data."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import structlog

from .models import (
    ConversationMessage,
    Finding,
    Checkpoint,
    RuleUpdate,
    ActionLog,
    SessionInfo,
)

logger = structlog.get_logger(__name__)


class SQLiteStore:
    """Async SQLite store for agent memory metadata and checkpoints."""

    def __init__(self, db_path: str = "data/agent_state.db"):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file (absolute path recommended)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = False

    async def initialize(self) -> None:
        """Initialize database schema if not already done."""
        if self._init_lock:
            return

        self._init_lock = True

        async with aiosqlite.connect(str(self.db_path)) as db:
            # Enable foreign keys
            await db.execute("PRAGMA foreign_keys = ON")

            # Sessions table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'active',
                    max_steps INTEGER NOT NULL DEFAULT 10,
                    current_step INTEGER NOT NULL DEFAULT 0,
                    findings_count INTEGER NOT NULL DEFAULT 0,
                    messages_count INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    total_latency_ms REAL NOT NULL DEFAULT 0.0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

            # Conversations table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tokens INTEGER NOT NULL,
                    model_used TEXT,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    UNIQUE(session_id, turn_number)
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)"
            )

            # Findings table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance INTEGER NOT NULL DEFAULT 5,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT NOT NULL DEFAULT 'general',
                    raw_text TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_findings_confidence ON findings(confidence)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_findings_importance ON findings(importance)"
            )

            # Checkpoints table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    agent_state TEXT NOT NULL,
                    memory_state TEXT NOT NULL,
                    completed_actions TEXT NOT NULL DEFAULT '[]',
                    next_actions TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    UNIQUE(session_id, step_number)
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id)"
            )

            # Rule updates table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS rule_updates (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    old_rule TEXT,
                    new_rule TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'proposed',
                    ab_test_score_old REAL,
                    ab_test_score_new REAL,
                    applied_at TIMESTAMP,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rule_updates_rule_id ON rule_updates(rule_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rule_updates_status ON rule_updates(status)"
            )

            # Action log table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS action_log (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action_type TEXT NOT NULL,
                    tool_name TEXT,
                    input_data TEXT NOT NULL DEFAULT '{}',
                    output_data TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    duration_ms REAL NOT NULL DEFAULT 0.0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_log_session ON action_log(session_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_log_timestamp ON action_log(timestamp)"
            )

            await db.commit()
            logger.info("sqlite_db_initialized", path=str(self.db_path))

    async def save_session(self, session: SessionInfo) -> None:
        """
        Save or update a session.

        Args:
            session: SessionInfo model to save
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                INSERT OR REPLACE INTO sessions
                (id, objective, created_at, updated_at, status, max_steps, current_step,
                 findings_count, messages_count, total_tokens, total_latency_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.objective,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.status,
                    session.max_steps,
                    session.current_step,
                    session.findings_count,
                    session.messages_count,
                    session.total_tokens,
                    session.total_latency_ms,
                    json.dumps(session.metadata),
                ),
            )
            await db.commit()
            logger.debug("session_saved", session_id=session.id)

    async def get_session(self, session_id: str) -> SessionInfo | None:
        """
        Get a session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            SessionInfo or None if not found
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()

            if not row:
                return None

            return SessionInfo(
                id=row["id"],
                objective=row["objective"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                status=row["status"],
                max_steps=row["max_steps"],
                current_step=row["current_step"],
                findings_count=row["findings_count"],
                messages_count=row["messages_count"],
                total_tokens=row["total_tokens"],
                total_latency_ms=row["total_latency_ms"],
                metadata=json.loads(row["metadata"]),
            )

    async def save_conversation(self, message: ConversationMessage) -> None:
        """
        Save a conversation message.

        Args:
            message: ConversationMessage to save
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                INSERT OR REPLACE INTO conversations
                (id, session_id, turn_number, role, content, timestamp, tokens,
                 model_used, latency_ms, tool_calls, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.turn_number,
                    message.role,
                    message.content,
                    message.timestamp.isoformat(),
                    message.tokens,
                    message.model_used,
                    message.latency_ms,
                    json.dumps(message.tool_calls),
                    json.dumps(message.metadata),
                ),
            )
            await db.commit()
            logger.debug("conversation_saved", message_id=message.id, session_id=message.session_id)

    async def get_conversation_history(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[ConversationMessage]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID
            limit: Maximum messages to return
            offset: Number of messages to skip

        Returns:
            List of ConversationMessage objects
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM conversations
                WHERE session_id = ?
                ORDER BY turn_number ASC
                LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            )
            rows = await cursor.fetchall()

            messages = []
            for row in rows:
                messages.append(
                    ConversationMessage(
                        id=row["id"],
                        session_id=row["session_id"],
                        turn_number=row["turn_number"],
                        role=row["role"],
                        content=row["content"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        tokens=row["tokens"],
                        model_used=row["model_used"],
                        latency_ms=row["latency_ms"],
                        tool_calls=json.loads(row["tool_calls"]),
                        metadata=json.loads(row["metadata"]),
                    )
                )

            return messages

    async def save_finding(self, finding: Finding) -> None:
        """
        Save a research finding.

        Args:
            finding: Finding to save
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                INSERT OR REPLACE INTO findings
                (id, session_id, title, content, sources, confidence, tags,
                 importance, timestamp, category, raw_text, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding.id,
                    finding.session_id,
                    finding.title,
                    finding.content,
                    json.dumps(finding.sources),
                    finding.confidence,
                    json.dumps(finding.tags),
                    finding.importance,
                    finding.timestamp.isoformat(),
                    finding.category,
                    finding.raw_text,
                    json.dumps(finding.metadata),
                ),
            )
            await db.commit()
            logger.debug("finding_saved", finding_id=finding.id, session_id=finding.session_id)

    async def get_findings(
        self, session_id: str, limit: int = 100, min_confidence: float = 0.0
    ) -> list[Finding]:
        """
        Get findings for a session.

        Args:
            session_id: Session ID
            limit: Maximum findings to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of Finding objects
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM findings
                WHERE session_id = ? AND confidence >= ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                (session_id, min_confidence, limit),
            )
            rows = await cursor.fetchall()

            findings = []
            for row in rows:
                findings.append(
                    Finding(
                        id=row["id"],
                        session_id=row["session_id"],
                        title=row["title"],
                        content=row["content"],
                        sources=json.loads(row["sources"]),
                        confidence=row["confidence"],
                        tags=json.loads(row["tags"]),
                        importance=row["importance"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        category=row["category"],
                        raw_text=row["raw_text"],
                        metadata=json.loads(row["metadata"]),
                    )
                )

            return findings

    async def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint.

        Args:
            checkpoint: Checkpoint to save
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (id, session_id, step_number, timestamp, agent_state, memory_state,
                 completed_actions, next_actions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.session_id,
                    checkpoint.step_number,
                    checkpoint.timestamp.isoformat(),
                    json.dumps(checkpoint.agent_state),
                    json.dumps(checkpoint.memory_state),
                    json.dumps(checkpoint.completed_actions),
                    json.dumps(checkpoint.next_actions),
                    json.dumps(checkpoint.metadata),
                ),
            )
            await db.commit()
            logger.debug("checkpoint_saved", checkpoint_id=checkpoint.id, session_id=checkpoint.session_id)

    async def get_latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        """
        Get the latest checkpoint for a session.

        Args:
            session_id: Session ID

        Returns:
            Latest Checkpoint or None if not found
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM checkpoints
                WHERE session_id = ?
                ORDER BY step_number DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return Checkpoint(
                id=row["id"],
                session_id=row["session_id"],
                step_number=row["step_number"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                agent_state=json.loads(row["agent_state"]),
                memory_state=json.loads(row["memory_state"]),
                completed_actions=json.loads(row["completed_actions"]),
                next_actions=json.loads(row["next_actions"]),
                metadata=json.loads(row["metadata"]),
            )

    async def save_rule_update(self, rule_update: RuleUpdate) -> None:
        """
        Save a rule update.

        Args:
            rule_update: RuleUpdate to save
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO rule_updates
                (id, rule_id, rule_type, old_rule, new_rule, reason, timestamp,
                 status, ab_test_score_old, ab_test_score_new, applied_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule_update.id,
                    rule_update.rule_id,
                    rule_update.rule_type,
                    rule_update.old_rule,
                    rule_update.new_rule,
                    rule_update.reason,
                    rule_update.timestamp.isoformat(),
                    rule_update.status,
                    rule_update.ab_test_score_old,
                    rule_update.ab_test_score_new,
                    rule_update.applied_at.isoformat() if rule_update.applied_at else None,
                    json.dumps(rule_update.metadata),
                ),
            )
            await db.commit()
            logger.debug("rule_update_saved", rule_update_id=rule_update.id)

    async def save_action(self, action: ActionLog) -> None:
        """
        Save an action log entry.

        Args:
            action: ActionLog to save
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                INSERT OR REPLACE INTO action_log
                (id, session_id, timestamp, action_type, tool_name, input_data,
                 output_data, status, error_message, duration_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.id,
                    action.session_id,
                    action.timestamp.isoformat(),
                    action.action_type,
                    action.tool_name,
                    json.dumps(action.input_data),
                    json.dumps(action.output_data),
                    action.status,
                    action.error_message,
                    action.duration_ms,
                    json.dumps(action.metadata),
                ),
            )
            await db.commit()
            logger.debug("action_logged", action_id=action.id, session_id=action.session_id)

    async def get_actions(
        self, session_id: str, limit: int = 100, status: str | None = None
    ) -> list[ActionLog]:
        """
        Get action logs for a session.

        Args:
            session_id: Session ID
            limit: Maximum actions to return
            status: Filter by status (pending, success, failed) or None for all

        Returns:
            List of ActionLog objects
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row

            if status:
                cursor = await db.execute(
                    """
                    SELECT * FROM action_log
                    WHERE session_id = ? AND status = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, status, limit),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT * FROM action_log
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )

            rows = await cursor.fetchall()

            actions = []
            for row in rows:
                actions.append(
                    ActionLog(
                        id=row["id"],
                        session_id=row["session_id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        action_type=row["action_type"],
                        tool_name=row["tool_name"],
                        input_data=json.loads(row["input_data"]),
                        output_data=json.loads(row["output_data"]),
                        status=row["status"],
                        error_message=row["error_message"],
                        duration_ms=row["duration_ms"],
                        metadata=json.loads(row["metadata"]),
                    )
                )

            return actions

    async def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """
        Get comprehensive statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session statistics
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row

            # Get conversation counts
            conv_cursor = await db.execute(
                "SELECT COUNT(*) as count, SUM(tokens) as total_tokens FROM conversations WHERE session_id = ?",
                (session_id,),
            )
            conv_row = await conv_cursor.fetchone()

            # Get findings counts
            find_cursor = await db.execute(
                "SELECT COUNT(*) as count, AVG(confidence) as avg_confidence FROM findings WHERE session_id = ?",
                (session_id,),
            )
            find_row = await find_cursor.fetchone()

            # Get action counts
            action_cursor = await db.execute(
                "SELECT COUNT(*) as count, SUM(duration_ms) as total_duration FROM action_log WHERE session_id = ?",
                (session_id,),
            )
            action_row = await action_cursor.fetchone()

            return {
                "session_id": session_id,
                "messages": conv_row["count"] or 0,
                "total_tokens": conv_row["total_tokens"] or 0,
                "findings": find_row["count"] or 0,
                "avg_confidence": find_row["avg_confidence"] or 0.0,
                "actions": action_row["count"] or 0,
                "total_action_duration_ms": action_row["total_duration"] or 0.0,
            }

    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days_old: Delete sessions older than this many days

        Returns:
            Number of sessions deleted
        """
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("PRAGMA foreign_keys = ON")

            # Get sessions to delete
            cursor = await db.execute(
                """
                SELECT id FROM sessions
                WHERE datetime(created_at) < datetime('now', ? || ' days')
                """,
                (f"-{days_old}",),
            )
            rows = await cursor.fetchall()
            session_ids = [row[0] for row in rows]

            # Delete cascades to related tables via foreign keys
            for session_id in session_ids:
                await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

            await db.commit()
            logger.info("sessions_cleaned", count=len(session_ids), days_old=days_old)
            return len(session_ids)

    async def close(self) -> None:
        """Close database connection."""
        # aiosqlite closes connections automatically, but explicit cleanup is good practice
        logger.debug("sqlite_store_closed")
