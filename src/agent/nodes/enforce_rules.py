"""Enforce rules node: apply Constitutional AI rules to final response."""

import asyncio
from datetime import datetime

import structlog

from src.llm.router import ModelRouter
from src.memory import MemoryManager
from ..state import AgentState

logger = structlog.get_logger(__name__)


async def enforce_rules_node(
    state: AgentState,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
) -> AgentState:
    """
    Apply Constitutional AI rules to the final synthesized response.

    Checks response against:
    - Hard rules (non-negotiable): e.g., "Verify claims against 2+ sources"
    - Soft rules (improvement suggestions): e.g., "Prefer primary sources"

    If violations found, sends response back for revision.

    Args:
        state: Agent state with final_response set
        memory_manager: Memory manager for rule persistence
        model_router: Model router for revision reasoning

    Returns:
        Updated state with rule_violations list and optionally revised response

    Example:
        Input: Synthesis response without proper citations
        Violations: ["H1: Claims not verified against 2 sources"]
        Revision: Response rewritten with proper citations
    """
    session_id = state.get("session_id", "unknown")
    final_response = state.get("final_response", "")

    logger.info("enforce_rules_start", session_id=session_id)

    if not final_response:
        logger.warning("no_response_to_enforce", session_id=session_id)
        state["rule_violations"] = []
        return state

    # For now, placeholder implementation
    # In full version, would:
    # 1. Load rules from config/rules.yaml
    # 2. Run each rule against response
    # 3. Collect violations
    # 4. Request revisions if needed

    violations = []

    # Mock rule checking
    mock_violations = _check_mock_rules(final_response)
    violations.extend(mock_violations)

    if violations:
        logger.warning("rule_violations_found", count=len(violations))

        # Attempt revision
        revised = await _revise_for_rules(
            final_response, violations, model_router
        )
        state["final_response"] = revised

    state["rule_violations"] = violations

    # Add to message history
    message = {
        "role": "system",
        "content": f"Applied {len(violations)} rule checks",
        "violations": violations,
        "timestamp": datetime.utcnow().isoformat(),
    }
    state["messages"].append(message)

    logger.info("enforce_rules_complete", violations_count=len(violations))

    return state


def _check_mock_rules(response: str) -> list[dict]:
    """
    Mock rule checking (placeholder).

    In production, would load rules from config/rules.yaml
    and run comprehensive checks.

    Args:
        response: Response text to check

    Returns:
        List of violation dicts
    """
    violations = []

    # Mock check 1: Response includes citations
    if "[" not in response or "]" not in response:
        violations.append({
            "rule_id": "H1",
            "rule_type": "hard",
            "rule": "Response should include citations in [N] format",
            "status": "violation",
        })

    # Mock check 2: Response length reasonable
    if len(response) < 100:
        violations.append({
            "rule_id": "S1",
            "rule_type": "soft",
            "rule": "Response should provide substantial information",
            "status": "warning",
        })

    return violations


async def _revise_for_rules(
    response: str,
    violations: list[dict],
    model_router: ModelRouter,
) -> str:
    """
    Revise response to address rule violations.

    Args:
        response: Original response
        violations: List of detected violations
        model_router: Model router for revision

    Returns:
        Revised response
    """
    violation_str = "\n".join(
        [f"- {v['rule_id']}: {v['rule']}" for v in violations]
    )

    system_prompt = """You are revising research responses to meet quality standards.
Address the specific violations noted below while maintaining the core information."""

    prompt = f"""Original response:
{response[:1000]}

Rule violations to address:
{violation_str}

Revise the response to address these violations. Keep the same core information but improve quality."""

    try:
        response_obj = await model_router.complete(
            role="synthesize",
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1500,
        )

        revised = response_obj.text.strip()
        logger.info("rule_revision_complete")
        return revised

    except Exception as e:
        logger.warning("rule_revision_failed", error=str(e))
        return response  # Return original if revision fails
