"""Reflect node: analyze progress and decide continuation."""

from datetime import datetime

import structlog

from src.llm.router import ModelRouter
from src.memory import MemoryManager
from ..state import AgentState

logger = structlog.get_logger(__name__)


def reflect_node(
    state: AgentState,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
) -> AgentState:
    """
    Analyze research progress and decide whether to continue or synthesize.

    Checks:
    1. Max steps limit (15) → force synthesis
    2. Context budget utilization (80% → trigger compression)
    3. Research completeness (more sub-goals to explore?)
    4. Information sufficiency (enough findings?)

    Decision logic:
    - If max_steps reached → synthesize
    - If context > 80% → compress memory, consider synthesis
    - If current_goal_index < len(sub_goals) → continue to next goal
    - If no more goals or goals exhausted → synthesize

    Args:
        state: Current agent state with findings and step count
        memory_manager: Memory manager for compression
        model_router: Not used in this node

    Returns:
        Updated state with should_continue flag set

    Example:
        Step 10/15, 4 findings, goal 2/3
        → should_continue = True, move to goal 3

        Step 15/15, max reached
        → should_continue = False, force synthesis
    """
    session_id = state.get("session_id", "unknown")
    step_number = state.get("step_number", 0)
    max_steps = state.get("max_steps", 15)
    current_goal_index = state.get("current_goal_index", 0)
    sub_goals = state.get("sub_goals", [])
    findings_count = len(state.get("findings", []))
    context_tokens = state.get("context_tokens", 0)
    max_context = state.get("max_context_tokens", 16384)

    logger.info(
        "reflect_start",
        session_id=session_id,
        step=step_number,
        max_steps=max_steps,
        goal=f"{current_goal_index + 1}/{len(sub_goals)}",
        findings=findings_count,
        context_utilization=f"{context_tokens / max_context:.1%}",
    )

    # Decision tree
    should_continue = True
    reason = "unknown"

    # 1. Check max steps limit (hard constraint)
    if step_number >= max_steps:
        should_continue = False
        reason = "max_steps_reached"
        logger.info("reflect_decision", decision="synthesize", reason=reason)

    # 2. Check context budget
    context_utilization = context_tokens / max_context
    if context_utilization >= 0.95:
        should_continue = False
        reason = "context_critical"
        logger.warning("context_critical_level", utilization=f"{context_utilization:.1%}")

    elif context_utilization >= 0.8:
        # Trigger memory compression
        logger.warning(
            "context_warning_level", utilization=f"{context_utilization:.1%}"
        )
        # Would call memory_manager.compress() here
        # For now, just log

    # 3. Check goal completion
    if should_continue and current_goal_index >= len(sub_goals):
        # All goals explored
        if findings_count < 3:
            # Not enough findings, try broader search
            logger.info("low_findings", count=findings_count, attempting_broader_search=True)
        else:
            should_continue = False
            reason = "all_goals_completed"
            logger.info("reflect_decision", decision="synthesize", reason=reason)

    # 4. Check if we should move to next goal
    if should_continue and current_goal_index < len(sub_goals):
        # Move to next goal
        state["current_goal_index"] = current_goal_index + 1
        state["step_number"] = step_number + 1
        reason = "next_goal"
        logger.info(
            "reflect_decision",
            decision="continue",
            reason=reason,
            next_goal_index=current_goal_index + 1,
        )

    elif should_continue:
        # No more goals, but not forcing synthesis yet
        should_continue = False
        reason = "no_more_goals"
        logger.info("reflect_decision", decision="synthesize", reason=reason)

    # Set decision flags
    state["should_continue"] = should_continue
    state["synthesis_complete"] = not should_continue

    # Add reflection to message history
    message = {
        "role": "assistant",
        "content": f"Reflection: {reason}",
        "reflection": {
            "step": step_number,
            "max_steps": max_steps,
            "goal_index": current_goal_index,
            "total_goals": len(sub_goals),
            "findings": findings_count,
            "context_utilization": f"{context_utilization:.1%}",
            "decision": "continue" if should_continue else "synthesize",
            "reason": reason,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    state["messages"].append(message)

    logger.info(
        "reflect_complete",
        decision="continue" if should_continue else "synthesize",
        reason=reason,
    )

    return state
