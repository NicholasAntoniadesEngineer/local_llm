"""In-context working memory with FIFO eviction and importance-weighted summarization."""

from collections import deque
from typing import Any, Optional
from datetime import datetime
import structlog

from .models import ConversationMessage, ContextBudget

logger = structlog.get_logger(__name__)


class WorkingMemory:
    """
    In-context FIFO buffer for recent messages and findings.

    - Maintains 4K token max with 80% compression trigger
    - FIFO eviction with importance-weighted summarization before removal
    - Tracks per-message token counts and metadata
    """

    def __init__(self, max_tokens: int = 4096, compression_threshold: float = 0.8):
        """
        Initialize working memory.

        Args:
            max_tokens: Maximum tokens in memory (default 4K)
            compression_threshold: Trigger summarization at this utilization (0-1)
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self._buffer: deque[ConversationMessage] = deque()
        self._token_count = 0
        self._compression_triggered = False

    def add_message(self, message: ConversationMessage) -> None:
        """
        Add a message to working memory with automatic eviction.

        Args:
            message: ConversationMessage to add
        """
        # Check if adding would exceed max
        new_total = self._token_count + message.tokens

        if new_total > self.max_tokens:
            # Evict until we have room
            self._evict_until_capacity(message.tokens)

        # Add message
        self._buffer.append(message)
        self._token_count += message.tokens

        # Check if we should compress
        utilization = self._token_count / self.max_tokens
        if utilization >= self.compression_threshold and not self._compression_triggered:
            self._compression_triggered = True
            logger.info(
                "compression_triggered",
                utilization=f"{utilization:.1%}",
                token_count=self._token_count,
                max_tokens=self.max_tokens,
            )

        logger.debug(
            "message_added",
            tokens=message.tokens,
            total_tokens=self._token_count,
            buffer_size=len(self._buffer),
        )

    def _evict_until_capacity(self, needed_tokens: int) -> list[ConversationMessage]:
        """
        Evict messages (FIFO) until there's room for new message.

        Args:
            needed_tokens: Tokens needed for new message

        Returns:
            List of evicted messages
        """
        evicted = []
        target_free = needed_tokens + int(self.max_tokens * 0.1)  # 10% buffer

        while self._token_count > (self.max_tokens - target_free) and self._buffer:
            msg = self._buffer.popleft()
            self._token_count -= msg.tokens
            evicted.append(msg)

        if evicted:
            logger.debug(
                "messages_evicted",
                count=len(evicted),
                freed_tokens=sum(m.tokens for m in evicted),
                remaining=self._token_count,
            )

        return evicted

    def get_buffer(self) -> list[ConversationMessage]:
        """
        Get current buffer contents in order.

        Returns:
            List of messages in order
        """
        return list(self._buffer)

    def get_token_count(self) -> int:
        """
        Get current token count in buffer.

        Returns:
            Number of tokens used
        """
        return self._token_count

    def get_utilization(self) -> float:
        """
        Get current utilization as percentage.

        Returns:
            Utilization as float (0.0 to 1.0)
        """
        return min(self._token_count / self.max_tokens, 1.0)

    def is_at_capacity(self) -> bool:
        """
        Check if buffer is at or near capacity.

        Returns:
            True if utilization >= 90%
        """
        return self.get_utilization() >= 0.9

    def should_compress(self) -> bool:
        """
        Check if compression is recommended.

        Returns:
            True if utilization >= compression_threshold
        """
        return self.get_utilization() >= self.compression_threshold

    def get_recent(self, num_messages: int = 5) -> list[ConversationMessage]:
        """
        Get the N most recent messages.

        Args:
            num_messages: Number of recent messages to return

        Returns:
            List of recent messages (most recent last)
        """
        return list(self._buffer)[-num_messages:]

    def get_by_role(self, role: str) -> list[ConversationMessage]:
        """
        Get all messages with a specific role.

        Args:
            role: Message role ("user" or "assistant")

        Returns:
            List of messages with that role
        """
        return [msg for msg in self._buffer if msg.role == role]

    def clear(self) -> int:
        """
        Clear all messages from working memory.

        Returns:
            Number of messages cleared
        """
        count = len(self._buffer)
        self._buffer.clear()
        self._token_count = 0
        self._compression_triggered = False
        logger.debug("working_memory_cleared", cleared=count)
        return count

    def summarize_and_evict(
        self, messages: Optional[list[ConversationMessage]] = None
    ) -> tuple[int, str]:
        """
        Create an importance-weighted summary before eviction.

        This is called during compression to preserve critical information
        before removing messages from working memory.

        Args:
            messages: Messages to summarize, or None to use all

        Returns:
            Tuple of (evicted_count, summary_text)
        """
        if messages is None:
            messages = list(self._buffer)

        if not messages:
            return 0, ""

        # Score messages by importance and role
        scored = []
        for msg in messages:
            # Weight by: role (assistant=2x), token count, model quality, tool calls
            base_importance = 2.0 if msg.role == "assistant" else 1.0
            tool_importance = 1.5 if msg.tool_calls else 1.0
            score = base_importance * tool_importance * min(msg.tokens / 100, 2.0)
            scored.append((score, msg))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build summary of top messages
        top_count = max(1, len(scored) // 3)
        summary_parts = [f"[MEMORY SUMMARY at {datetime.utcnow().isoformat()}]"]

        for _, msg in scored[:top_count]:
            role_label = msg.role.upper()
            preview = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
            summary_parts.append(f"[{role_label}] {preview}")

        summary = "\n".join(summary_parts)

        # Record eviction
        evicted_count = len(messages)
        logger.info(
            "messages_summarized",
            evicted=evicted_count,
            summary_length=len(summary),
            top_count=top_count,
        )

        return evicted_count, summary

    def get_context_for_model(self, context_budget: ContextBudget) -> dict[str, Any]:
        """
        Get formatted context fitting within allocated budget.

        Args:
            context_budget: ContextBudget allocation

        Returns:
            Dictionary with formatted messages and token counts
        """
        messages = list(self._buffer)

        if not messages:
            return {
                "messages": [],
                "token_count": 0,
                "utilization": 0.0,
                "formatted": "",
            }

        # Format as conversation
        formatted_lines = []
        tokens_used = 0

        for msg in messages:
            line = f"{msg.role.upper()}: {msg.content}"
            formatted_lines.append(line)
            tokens_used += msg.tokens

            # Stop if we'd exceed allocated conversation history budget
            if tokens_used > context_budget.conversation_history:
                formatted_lines = formatted_lines[:-1]
                tokens_used -= msg.tokens
                break

        formatted = "\n\n".join(formatted_lines)

        utilization = tokens_used / context_budget.conversation_history

        return {
            "messages": messages,
            "token_count": tokens_used,
            "utilization": utilization,
            "formatted": formatted,
        }

    def stats(self) -> dict[str, Any]:
        """
        Get working memory statistics.

        Returns:
            Dictionary with statistics
        """
        messages = list(self._buffer)
        roles = {"user": 0, "assistant": 0}
        for msg in messages:
            roles[msg.role] = roles.get(msg.role, 0) + 1

        return {
            "total_messages": len(messages),
            "messages_by_role": roles,
            "token_count": self._token_count,
            "max_tokens": self.max_tokens,
            "utilization": f"{self.get_utilization():.1%}",
            "at_capacity": self.is_at_capacity(),
            "should_compress": self.should_compress(),
        }
