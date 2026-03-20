"""Agent state definition for LangGraph orchestration."""

from typing import TypedDict, Any, Literal, Optional
from datetime import datetime


class AgentState(TypedDict, total=False):
    """
    Complete agent state for LangGraph execution.

    Tracks research objectives, sub-goals, findings, memory usage,
    and execution history for resumability and learning.

    Example:
        state = AgentState(
            objective="Find latest advances in quantum computing",
            session_id="sess_abc123",
            step_number=0,
            max_steps=15,
            findings=[],
            messages=[],
        )
    """

    # Core research task
    objective: str
    """User's research objective/query."""

    session_id: str
    """Unique session identifier for resumability."""

    # Planning and decomposition
    sub_goals: list[str]
    """Decomposed sub-goals from objective (max 5)."""

    current_goal_index: int
    """Index of currently active sub-goal (0-based)."""

    # Execution tracking
    step_number: int
    """Current step in research loop (0-based)."""

    max_steps: int
    """Maximum steps allowed before forcing synthesis."""

    # Research findings
    findings: list[dict[str, Any]]
    """Accumulated research findings with metadata."""

    # Conversation history
    messages: list[dict[str, Any]]
    """Turn-by-turn conversation history for context."""

    # Memory and context budget
    context_tokens: int
    """Current token usage in context window."""

    max_context_tokens: int
    """Maximum tokens allowed (16384)."""

    # Rules and quality control
    rule_violations: list[dict[str, Any]]
    """Detected rule violations from last round."""

    proposed_rule_changes: list[dict[str, Any]]
    """Agent-proposed rule improvements for A/B testing."""

    # Tool execution results
    last_tool_results: dict[str, Any]
    """Results from last tool invocation batch."""

    tool_error: Optional[str]
    """Error message if tool execution failed."""

    # Execution metadata
    checkpoint_data: dict[str, Any]
    """Last checkpoint for resumability."""

    created_at: datetime
    """Session start timestamp."""

    updated_at: datetime
    """Last update timestamp."""

    should_continue: bool
    """Flag to continue research loop or move to synthesis."""

    synthesis_complete: bool
    """Flag indicating synthesis phase completed."""

    final_response: str
    """Final synthesized response (set in synthesis node)."""

    error_message: Optional[str]
    """Error message if execution failed."""

    execution_status: Literal[
        "planning", "researching", "reflecting", "synthesizing", "enforcing_rules", "complete", "failed"
    ]
    """Current execution phase."""

    parent_message_id: Optional[str]
    """Parent message ID for conversation threading (multi-turn support)."""

    metadata: dict[str, Any]
    """Additional metadata (tags, user_id, source, etc.)."""
