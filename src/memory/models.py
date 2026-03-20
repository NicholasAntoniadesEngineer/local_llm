"""Pydantic v2 models for all memory types."""

from typing import Any, Literal, Optional
from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field, field_validator, ConfigDict


class MemoryEntryType(str, Enum):
    """Types of memory entries."""
    CONVERSATION = "conversation"
    FINDING = "finding"
    CHECKPOINT = "checkpoint"
    ACTION = "action"
    RULE_UPDATE = "rule_update"
    SESSION = "session"


class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    turn_number: int
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tokens: int
    model_used: str | None = None
    latency_ms: float = 0.0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("turn_number")
    @classmethod
    def validate_turn_number(cls, v: int) -> int:
        """Ensure turn number is non-negative."""
        if v < 0:
            raise ValueError("turn_number must be non-negative")
        return v

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, v: int) -> int:
        """Ensure tokens is non-negative."""
        if v < 0:
            raise ValueError("tokens must be non-negative")
        return v

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, v: float) -> float:
        """Ensure latency is non-negative."""
        if v < 0:
            raise ValueError("latency_ms must be non-negative")
        return v


class Finding(BaseModel):
    """A research finding or fact."""
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    title: str
    content: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(ge=1, le=10, default=5)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    category: str = "general"
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        return v


class Checkpoint(BaseModel):
    """A session checkpoint for resumability."""
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    step_number: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_state: dict[str, Any]
    memory_state: dict[str, Any]
    completed_actions: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("step_number")
    @classmethod
    def validate_step_number(cls, v: int) -> int:
        """Ensure step number is non-negative."""
        if v < 0:
            raise ValueError("step_number must be non-negative")
        return v


class RuleUpdate(BaseModel):
    """A proposed or accepted rule change."""
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str
    rule_type: Literal["hard", "soft"]
    old_rule: str | None = None
    new_rule: str
    reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["proposed", "accepted", "rejected", "reverted"] = "proposed"
    ab_test_score_old: float | None = None
    ab_test_score_new: float | None = None
    applied_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ab_test_score_old", "ab_test_score_new")
    @classmethod
    def validate_scores(cls, v: float | None) -> float | None:
        """Ensure scores are between 0 and 1 if provided."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("A/B test scores must be between 0 and 1")
        return v


class ActionLog(BaseModel):
    """A log entry for an agent action."""
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action_type: str
    tool_name: str | None = None
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "success", "failed"] = "pending"
    error_message: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("duration_ms")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        """Ensure duration is non-negative."""
        if v < 0:
            raise ValueError("duration_ms must be non-negative")
        return v


class SessionInfo(BaseModel):
    """Information about a research session."""
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["active", "paused", "completed", "failed"] = "active"
    max_steps: int = 10
    current_step: int = 0
    findings_count: int = 0
    messages_count: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("max_steps", "current_step", "findings_count", "messages_count", "total_tokens")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Ensure counters are non-negative."""
        if v < 0:
            raise ValueError("Counter values must be non-negative")
        return v


class ContextBudget(BaseModel):
    """Context window budget allocation for 16K window."""
    model_config = ConfigDict(validate_assignment=True)

    system_prompt: int = 1024
    tool_definitions: int = 2048
    retrieved_memory: int = 4096
    conversation_history: int = 4096
    workspace_scratch: int = 3072
    response_buffer: int = 2048

    # Derived properties
    total_budget: int = 16384
    warning_threshold: float = 0.8  # 80% full
    critical_threshold: float = 0.95  # 95% full

    def get_total_allocated(self) -> int:
        """Get total allocated tokens."""
        return (
            self.system_prompt
            + self.tool_definitions
            + self.retrieved_memory
            + self.conversation_history
            + self.workspace_scratch
            + self.response_buffer
        )

    def get_utilization(self, used_tokens: int) -> float:
        """Get current utilization as percentage."""
        return min(used_tokens / self.total_budget, 1.0)

    def is_warning_level(self, used_tokens: int) -> bool:
        """Check if utilization exceeds warning threshold."""
        return self.get_utilization(used_tokens) >= self.warning_threshold

    def is_critical_level(self, used_tokens: int) -> bool:
        """Check if utilization exceeds critical threshold."""
        return self.get_utilization(used_tokens) >= self.critical_threshold

    def remaining_budget(self, used_tokens: int) -> int:
        """Get remaining tokens available."""
        return max(0, self.total_budget - used_tokens)


class MemoryEntry(BaseModel):
    """Union type for all memory entries."""
    model_config = ConfigDict(validate_assignment=True)

    entry_type: MemoryEntryType
    entry_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
